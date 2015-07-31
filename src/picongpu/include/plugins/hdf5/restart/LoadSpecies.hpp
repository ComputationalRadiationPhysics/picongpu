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
#include <boost/type_traits.hpp>

#include "compileTime/conversion/MakeSeq.hpp"
#include "compileTime/conversion/RemoveFromSeq.hpp"
#include "mappings/kernel/AreaMapping.hpp"
#include "particles/ParticleDescription.hpp"

#include "plugins/output/WriteSpeciesCommon.hpp"
#include "plugins/kernel/CopySpeciesGlobal2Local.kernel"
#include "plugins/hdf5/restart/LoadParticleAttributesFromHDF5.hpp"

namespace picongpu
{

namespace hdf5
{
using namespace PMacc;

using namespace splash;

/** Load species from HDF5 checkpoint file
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

    /** Load species from HDF5 checkpoint file
     *
     * @param params thread params with domainwriter, ...
     * @param restartChunkSize number of particles processed in one kernel call
     */
    HINLINE void operator()(ThreadParams* params, const uint32_t restartChunkSize)
    {

        log<picLog::INPUT_OUTPUT > ("HDF5: (begin) load species: %1%") % Hdf5FrameType::getName();
        DataConnector &dc = Environment<>::get().DataConnector();
        GridController<simDim> &gc = Environment<simDim>::get().GridController();

        std::string subGroup = std::string("particles/") + FrameType::getName();
        const PMacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();

        /* load particle without copying particle data to host */
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

        Hdf5FrameType hostFrame;
        log<picLog::INPUT_OUTPUT > ("HDF5:  malloc mapped memory: %1%") % Hdf5FrameType::getName();
        /*malloc mapped memory*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, MallocMemory<bmpl::_1> > mallocMem;
        mallocMem(forward(hostFrame), totalNumParticles);

        log<picLog::INPUT_OUTPUT > ("HDF5:  get mapped memory device pointer: %1%") % Hdf5FrameType::getName();
        /*load device pointer of mapped memory*/
        Hdf5FrameType deviceFrame;
        ForEach<typename Hdf5FrameType::ValueTypeSeq, GetDevicePtr<bmpl::_1> > getDevicePtr;
        getDevicePtr(forward(deviceFrame), forward(hostFrame));

        ForEach<typename Hdf5FrameType::ValueTypeSeq, LoadParticleAttributesFromHDF5<bmpl::_1> > loadAttributes;
        loadAttributes(forward(params), forward(hostFrame), subGroup, particleOffset, totalNumParticles);

        if (totalNumParticles != 0)
        {
            dim3 block(PMacc::math::CT::volume<SuperCellSize>::type::value);

            /* counter is used to apply for work, count used frames and count loaded particles
             * [0] -> offset for loading particles
             * [1] -> number of loaded particles
             * [2] -> number of used frames
             *
             * all values are zero after initialization
             */
            GridBuffer<uint32_t, DIM1> counterBuffer(DataSpace<DIM1>(3));

            const uint32_t cellsInSuperCell = PMacc::math::CT::volume<SuperCellSize>::type::value;

            const uint32_t iterationsForLoad = ceil(float_64(totalNumParticles) / float_64(restartChunkSize));
            uint32_t leftOverParticles = totalNumParticles;

            __startAtomicTransaction(__getTransactionEvent());

            for (uint32_t i = 0; i < iterationsForLoad; ++i)
            {
                /* only load a chunk of particles per iteration to avoid blow up of frame usage
                 */
                uint32_t currentChunkSize = std::min(leftOverParticles, restartChunkSize);
                log<picLog::INPUT_OUTPUT > ("HDF5:   load particles on device chunk offset=%1%; chunk size=%2%; left particles %3%") %
                    (i * restartChunkSize) % currentChunkSize % leftOverParticles;
                __cudaKernel(copySpeciesGlobal2Local)
                    (ceil(float_64(currentChunkSize) / float_64(cellsInSuperCell)), cellsInSuperCell)
                    (counterBuffer.getDeviceBuffer().getDataBox(),
                     speciesTmp->getDeviceParticlesBox(), deviceFrame,
                     (int) totalNumParticles,
                     localDomain.offset, /*relative to data domain (not to physical domain)*/
                     *(params->cellDescription)
                     );
                speciesTmp->fillAllGaps();
                leftOverParticles -= currentChunkSize;
            }
            __setTransactionEvent(__endTransaction());
            counterBuffer.deviceToHost();
            log<picLog::INPUT_OUTPUT > ("HDF5:  wait for last processed chunk: %1%") % Hdf5FrameType::getName();
            __getTransactionEvent().waitForFinished();

            log<picLog::INPUT_OUTPUT > ("HDF5: used frames to load particles: %1%") % counterBuffer.getHostBuffer().getDataBox()[2];

            if ((uint64_cu) counterBuffer.getHostBuffer().getDataBox()[1] != totalNumParticles)
            {
                log<picLog::INPUT_OUTPUT >("HDF5:  error load species | counter is %1% but should %2%") % counterBuffer.getHostBuffer().getDataBox()[1] % totalNumParticles;
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
