/**
 * Copyright 2013-2015 Rene Widera, Felix Schmitt, Axel Huebl
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

#include "plugins/adios/ADIOSWriter.def"
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
#include "plugins/adios/restart/LoadParticleAttributesFromADIOS.hpp"

namespace picongpu
{

namespace adios
{
using namespace PMacc;

/** Load species from ADIOS checkpoint file
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


    /* delete multiMask and localCellIdx in adios particle*/
    typedef bmpl::vector2<multiMask, localCellIdx> TypesToDelete;
    typedef typename RemoveFromSeq<ParticleAttributeList, TypesToDelete>::type ParticleCleanedAttributeList;

    /* add globalCellIdx for adios particle*/
    typedef typename MakeSeq<
    ParticleCleanedAttributeList,
    globalCellIdx<globalCellIdx_pic>
    >::type ParticleNewAttributeList;

    typedef
    typename ReplaceValueTypeSeq<ParticleDescription, ParticleNewAttributeList>::type
    NewParticleDescription;

    typedef Frame<OperatorCreateVectorBox, NewParticleDescription> AdiosFrameType;

    /** Load species from ADIOS checkpoint file
     *
     * @param params thread params with ADIOS_FILE, ...
     * @param restartChunkSize number of particles processed in one kernel call
     */
    HINLINE void operator()(ThreadParams* params, const uint32_t restartChunkSize)
    {

        log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) load species: %1%") % AdiosFrameType::getName();
        DataConnector &dc = Environment<>::get().DataConnector();
        GridController<simDim> &gc = Environment<simDim>::get().GridController();

        std::string particlePath = params->adiosBasePath + std::string(ADIOS_PATH_PARTICLES) +
                                   FrameType::getName() + std::string("/");
        const PMacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();

        /* load particle without copying particle data to host */
        ThisSpecies* speciesTmp = &(dc.getData<ThisSpecies >(ThisSpecies::FrameType::getName(), true));

        /* count total number of particles on the device */
        uint64_t totalNumParticles = 0;

        /* load particles info table entry for ONE process
           (note: this is NOT necessarily THIS process!)
           particlesInfo is (part-count, scalar pos, x, y, z) */
        uint64_t particlesInfo[5];

        uint64_t start = 5 * gc.getGlobalRank();
        uint64_t count = 5; // ADIOSCountParticles: uint64_t
        ADIOS_SELECTION* piSel = adios_selection_boundingbox( 1, &start, &count );

        ADIOS_CMD(adios_schedule_read( params->fp,
                                       piSel,
                                       (particlePath + std::string("particles_info")).c_str(),
                                       0,
                                       1,
                                       (void*)particlesInfo ));

        /* start a blocking read of all scheduled variables */
        ADIOS_CMD(adios_perform_reads( params->fp, 1 ));
        adios_selection_delete(piSel);

        /* Run a prefix sum over the numParticles[0] element in particlesInfo
         * to retreive the offset of particles before gc.getGlobalRank() */
        uint64_t particleOffset = 0;

        uint64_t fullParticlesInfo[gc.getGlobalSize()];

        MPI_CHECK(MPI_Allgather( particlesInfo, 1, MPI_UINT64_T,
                                 fullParticlesInfo, 1, MPI_UINT64_T,
                                 gc.getCommunicator().getMPIComm() ));

        for (size_t i = 0; i < gc.getGlobalSize(); ++i)
        {
            /* this comparison is potentially harmful, since the order of ranks
               is not necessarily the same in subsequent MPI jobs.
               But due to the wrong sorting by rank in `ADIOSCountParticles.hpp`
               while calculating the `myParticleOffset` we have to immitate that. */
            if( i < gc.getGlobalRank() )
                particleOffset += fullParticlesInfo[i];
            if( i == gc.getGlobalRank() )
                totalNumParticles = fullParticlesInfo[i];
        }

        log<picLog::INPUT_OUTPUT > ("ADIOS: Loading %1% particles from offset %2%") %
            (long long unsigned) totalNumParticles % (long long unsigned) particleOffset;

        AdiosFrameType hostFrame;
        log<picLog::INPUT_OUTPUT > ("ADIOS: malloc mapped memory: %1%") % AdiosFrameType::getName();
        /*malloc mapped memory*/
        ForEach<typename AdiosFrameType::ValueTypeSeq, MallocMemory<bmpl::_1> > mallocMem;
        mallocMem(forward(hostFrame), totalNumParticles);

        log<picLog::INPUT_OUTPUT > ("ADIOS: get mapped memory device pointer: %1%") % AdiosFrameType::getName();
        /*load device pointer of mapped memory*/
        AdiosFrameType deviceFrame;
        ForEach<typename AdiosFrameType::ValueTypeSeq, GetDevicePtr<bmpl::_1> > getDevicePtr;
        getDevicePtr(forward(deviceFrame), forward(hostFrame));

        ForEach<typename AdiosFrameType::ValueTypeSeq, LoadParticleAttributesFromADIOS<bmpl::_1> > loadAttributes;
        loadAttributes(forward(params), forward(hostFrame), particlePath, particleOffset, totalNumParticles);

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
                log<picLog::INPUT_OUTPUT > ("ADIOS: load particles on device chunk offset=%1%; chunk size=%2%; left particles %3%") %
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
            log<picLog::INPUT_OUTPUT > ("ADIOS: wait for last processed chunk: %1%") % AdiosFrameType::getName();
            __getTransactionEvent().waitForFinished();

            log<picLog::INPUT_OUTPUT > ("ADIOS: used frames to load particles: %1%") % counterBuffer.getHostBuffer().getDataBox()[2];

            if ((uint64_t) counterBuffer.getHostBuffer().getDataBox()[1] != totalNumParticles)
            {
                log<picLog::INPUT_OUTPUT >("ADIOS: error load species | counter is %1% but should %2%") % counterBuffer.getHostBuffer().getDataBox()[1] % totalNumParticles;
                throw std::runtime_error("ADIOS: Failed to load expected number of particles to GPU.");
            }
            assert((uint64_t) counterBuffer.getHostBuffer().getDataBox()[1] == totalNumParticles);

            /*free host memory*/
            ForEach<typename AdiosFrameType::ValueTypeSeq, FreeMemory<bmpl::_1> > freeMem;
            freeMem(forward(hostFrame));
        }
        log<picLog::INPUT_OUTPUT > ("ADIOS: ( end ) load species: %1%") % AdiosFrameType::getName();
    }
};


} /* namespace adios */

} /* namespace picongpu */
