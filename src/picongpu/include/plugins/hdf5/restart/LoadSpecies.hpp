/**
 * Copyright 2013-2014 Rene Widera, Felix Schmitt
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */



#pragma once


#include "types.h"
#include "simulation_types.hpp"
#include "plugins/hdf5/HDF5Writer.def"

#include "plugins/ISimulationPlugin.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>
#include "compileTime/conversion/MakeSeq.hpp"

#include <boost/type_traits.hpp>

#include "plugins/output/WriteSpeciesCommon.hpp"
#include "plugins/kernel/CopySpecies.kernel"
#include "mappings/kernel/AreaMapping.hpp"

#include "plugins/hdf5/writer/ParticleAttribute.hpp"
#include "compileTime/conversion/RemoveFromSeq.hpp"
#include "particles/ParticleDescription.hpp"

#include "plugins/kernel/CopySpeciesGlobal2Local.kernel"
#include "plugins/hdf5/restart/LoadParticleAttributesFromHDF5.hpp"

namespace picongpu
{

namespace hdf5
{
using namespace PMacc;

using namespace splash;

/** Load particle from HDF5 checkpoint file
 *
 * @tparam T_Species type of species
 *
 */
template< typename T_Species >
struct LoadSpecies
{
public:

    typedef T_Species ThisSpecies;
    typedef typename ThisSpecies::FrameType FrameType;
    typedef typename FrameType::ParticleDescription ParticleDescription;
    typedef typename FrameType::ValueTypeSeq ParticleAttributeList;


    /* delete multiMask and localCellIdx in hdf5 particle*/
    typedef bmpl::vector2<multiMask, localCellIdx> TypesToDelete;
    typedef typename RemoveFromSeq<ParticleAttributeList, TypesToDelete>::type ParticleCleanedAttributeList;

    /* add globalCellIdx for hdf5 particle*/
    typedef typename MakeSeq<
    ParticleCleanedAttributeList,
    globalCellIdx<globalCellIdx_pic>
    >::type ParticleNewAttributeList;

    typedef
    typename ReplaceValueTypeSeq<ParticleDescription, ParticleNewAttributeList>::type
    NewParticleDescription;

    typedef Frame<OperatorCreateVectorBox, NewParticleDescription> Hdf5FrameType;

    HINLINE void operator()(ThreadParams* params, const uint32_t &restartChunkSize)
    {

        log<picLog::INPUT_OUTPUT > ("HDF5: (begin) load species: %1%") % Hdf5FrameType::getName();
        DataConnector &dc = Environment<>::get().DataConnector();
        GridController<simDim> &gc = Environment<simDim>::get().GridController();

        std::string subGroup = std::string("particles/") + FrameType::getName();
        const DomainInformation domInfo;
        const DataSpace<simDim> gridOffset = domInfo.localDomain.offset;

        /* load particle without copy particle data to host */
        ThisSpecies* speciesTmp = &(dc.getData<ThisSpecies >(ThisSpecies::FrameType::getName(), true));

        /* count total number of particles on the device */
        uint64_cu totalNumParticles = 0;

        /* load particles info table entry for this process
           particlesInfo is (part-count, scalar pos, x, y, z) */
        typedef uint64_t uint64Quint[5];
        uint64Quint particlesInfo[gc.getGlobalSize()];
        Dimensions particlesInfoSizeRead;

        params->dataCollector->read(params->currentStep,
                                    (std::string(subGroup) + std::string("/particles_info")).c_str(),
                                    particlesInfoSizeRead,
                                    particlesInfo);

        assert(particlesInfoSizeRead[0] == gc.getGlobalSize());

        /* search my entry (using my scalar position) in particlesInfo */
        uint64_t particleOffset = 0;
        uint64_t myScalarPos = gc.getScalarPosition();

        for (size_t i = 0; i < particlesInfoSizeRead[0]; ++i)
        {
            if (particlesInfo[i][1] == myScalarPos)
            {
                totalNumParticles = particlesInfo[i][0];
                break;
            }

            particleOffset += particlesInfo[i][0];
        }

        log<picLog::INPUT_OUTPUT > ("Loading %1% particles from offset %2%") %
            (long long unsigned) totalNumParticles % (long long unsigned) particleOffset;

        if (totalNumParticles != 0)
        {

            Hdf5FrameType hostFrame;
            log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) malloc mapped memory: %1%") % Hdf5FrameType::getName();
            /*malloc mapped memory*/
            ForEach<typename Hdf5FrameType::ValueTypeSeq, MallocMemory<bmpl::_1> > mallocMem;
            mallocMem(forward(hostFrame), totalNumParticles);
            log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) malloc mapped memory: %1%") % Hdf5FrameType::getName();

            log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) get mapped memory device pointer: %1%") % Hdf5FrameType::getName();
            /*load device pointer of mapped memory*/
            Hdf5FrameType deviceFrame;
            ForEach<typename Hdf5FrameType::ValueTypeSeq, GetDevicePtr<bmpl::_1> > getDevicePtr;
            getDevicePtr(forward(deviceFrame), forward(hostFrame));
            log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) get mapped memory device pointer: %1%") % Hdf5FrameType::getName();

            ForEach<typename Hdf5FrameType::ValueTypeSeq, LoadParticleAttributesFromHDF5<bmpl::_1> > loadAttributes;
            loadAttributes(forward(params), forward(hostFrame), subGroup, particleOffset, totalNumParticles);

            dim3 block(PMacc::math::CT::volume<SuperCellSize>::type::value);

            GridBuffer<int, DIM1> counterBuffer(DataSpace<DIM1>(3));
            AreaMapping < CORE + BORDER, MappingDesc > mapper(*(params->cellDescription));

            log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) copy particle to device: %1%") % Hdf5FrameType::getName();

            const int cellsInSuperCell = PMacc::math::CT::volume<SuperCellSize>::type::value;

            const int iterationsForLoad = ceil(double(totalNumParticles) / double(restartChunkSize));
            int leftOverParticles = totalNumParticles;

            __startAtomicTransaction(__getTransactionEvent());

            for (int i = 0; i < iterationsForLoad; ++i)
            {
                /* only load a chunk of particles per iteration to avoid blow up of frame usage
                 */
                int currentChunkSize = std::min((int) leftOverParticles, (int) restartChunkSize);
                log<picLog::INPUT_OUTPUT > ("HDF5:   ( begin ) Load particle chunk offset=%1%; chunk size=%2%; left particles %3%") %
                    (i * restartChunkSize) % currentChunkSize % leftOverParticles;
                __cudaKernel(copySpeciesGlobal2Local)
                    (ceil(double(currentChunkSize) / double(cellsInSuperCell)), cellsInSuperCell)
                    (counterBuffer.getDeviceBuffer().getDataBox(),
                     speciesTmp->getDeviceParticlesBox(), deviceFrame,
                     (int) totalNumParticles,
                     gridOffset, /*relative to data domain (not to physical domain)*/
                     mapper
                     );
                speciesTmp->fillAllGaps();
                log<picLog::INPUT_OUTPUT > ("HDF5:   ( end ) Load particle chunk offset=%1%; chunk size=%2%; left particles %3%") %
                    (i * restartChunkSize) % currentChunkSize % leftOverParticles;
                leftOverParticles -= currentChunkSize;
            }
            __setTransactionEvent(__endTransaction());
            counterBuffer.deviceToHost();
            log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) copy particle to device: %1%") % Hdf5FrameType::getName();
            __getTransactionEvent().waitForFinished();

            log<picLog::INPUT_OUTPUT > ("HDF5: used frames to load particles: %1%") % counterBuffer.getHostBuffer().getDataBox()[2];

            if ((uint64_cu) counterBuffer.getHostBuffer().getDataBox()[1] != totalNumParticles)
            {
                std::cout << "counter=" << counterBuffer.getHostBuffer().getDataBox()[3] << " should=" << totalNumParticles << std::endl;
            }
            assert((uint64_cu) counterBuffer.getHostBuffer().getDataBox()[1] == totalNumParticles);

            /*free host memory*/
            ForEach<typename Hdf5FrameType::ValueTypeSeq, FreeMemory<bmpl::_1> > freeMem;
            freeMem(forward(hostFrame));
            log<picLog::INPUT_OUTPUT > ("HDF5: ( end ) load species: %1%") % Hdf5FrameType::getName();
        }
    }
};


} //namspace hdf5

} //namespace picongpu
