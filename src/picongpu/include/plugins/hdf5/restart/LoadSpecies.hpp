/**
 * Copyright 2013-2016 Rene Widera, Felix Schmitt
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

#include "plugins/common/particlePatches.hpp"
#include "plugins/hdf5/openPMD/patchReader.hpp"

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

        const std::string speciesSubGroup(
            std::string("particles/") + FrameType::getName() + std::string("/")
        );
        const PMacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();

        // load particle without copying particle data to host
        ThisSpecies* speciesTmp = &(dc.getData<ThisSpecies >(ThisSpecies::FrameType::getName(), true));

        // count total number of particles on the device
        uint64_cu totalNumParticles = 0;
        uint64_t particleOffset = 0;

        // load particle patches offsets to find own patch
        const std::string particlePatchesPath(
            speciesSubGroup + std::string("particlePatches/")
        );

        // read particle patches
        openPMD::PatchReader patchReader;

        picongpu::openPMD::ParticlePatches particlePatches(
            patchReader(
                params->dataCollector,
                gc.getGlobalSize(),
                simDim,
                params->currentStep,
                particlePatchesPath
            )
        );

        /** search my entry (using my cell offset and my local grid size)
         *
         * \note if you want to restart with a changed GPU configuration, either
         * post-process the particle-patches in the file or implement to find
         * all contributing patches and then filter the particles inside those
         * by position
         *
         * \see plugins/hdf5/WriteSpecies.hpp `WriteSpecies::operator()`
         *      as its counterpart
         */
        const DataSpace<simDim> patchOffset =
            params->window.globalDimensions.offset +
            params->window.localDimensions.offset +
            params->localWindowToDomainOffset;
        const DataSpace<simDim> patchExtent =
            params->window.localDimensions.size;

        for( size_t i = 0; i < gc.getGlobalSize(); ++i )
        {
            bool exactlyMyPatch = true;

            for( uint32_t d = 0; d < simDim; ++d )
            {
                if( particlePatches.getOffsetComp( d )[ i ] != (uint64_t)patchOffset[ d ] )
                    exactlyMyPatch = false;
                if( particlePatches.getExtentComp( d )[ i ] != (uint64_t)patchExtent[ d ] )
                    exactlyMyPatch = false;
            }

            if( exactlyMyPatch )
            {
                totalNumParticles = particlePatches.numParticles[ i ];
                particleOffset = particlePatches.numParticlesOffset[ i ];
                break;
            }
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
        loadAttributes(forward(params), forward(hostFrame), speciesSubGroup, particleOffset, totalNumParticles);

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
