/* Copyright 2013-2021 Rene Widera, Felix Schmitt, Axel Huebl
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/plugins/adios/ADIOSWriter.def"
#include "picongpu/plugins/ISimulationPlugin.hpp"

#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/meta/conversion/RemoveFromSeq.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/particles/ParticleDescription.hpp>
#include <pmacc/particles/operations/splitIntoListOfFrames.kernel>

#include "picongpu/plugins/output/WriteSpeciesCommon.hpp"
#include "picongpu/plugins/adios/restart/LoadParticleAttributesFromADIOS.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>

#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>
#include <boost/type_traits.hpp>


namespace picongpu
{
    namespace adios
    {
        using namespace pmacc;

        /** Load species from ADIOS checkpoint file
         *
         * @tparam T_Species type of species
         *
         */
        template<typename T_Species>
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

            /* add totalCellIdx for adios particle*/
            typedef typename MakeSeq<ParticleCleanedAttributeList, totalCellIdx>::type ParticleNewAttributeList;

            typedef typename ReplaceValueTypeSeq<ParticleDescription, ParticleNewAttributeList>::type
                NewParticleDescription;

            typedef Frame<OperatorCreateVectorBox, NewParticleDescription> AdiosFrameType;

            /** Load species from ADIOS checkpoint file
             *
             * @param params thread params with ADIOS_FILE, ...
             * @param restartChunkSize number of particles processed in one kernel call
             */
            HINLINE void operator()(ThreadParams* params, const uint32_t restartChunkSize)
            {
                std::string const speciesName = FrameType::getName();
                log<picLog::INPUT_OUTPUT>("ADIOS: (begin) load species: %1%") % speciesName;
                DataConnector& dc = Environment<>::get().DataConnector();
                GridController<simDim>& gc = Environment<simDim>::get().GridController();

                std::string particlePath
                    = params->adiosBasePath + std::string(ADIOS_PATH_PARTICLES) + speciesName + std::string("/");
                const pmacc::Selection<simDim> localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();

                /* load particle without copying particle data to host */
                auto speciesTmp = dc.get<ThisSpecies>(FrameType::getName(), true);

                /* count total number of particles on the device */
                uint64_t totalNumParticles = 0;

                /* load particles info table entry for ONE process
                   (note: this is NOT necessarily THIS process!)
                   particlesInfo is (part-count, scalar pos, x, y, z) */
                uint64_t particlesInfo[5];

                uint64_t start = 5 * gc.getGlobalRank();
                uint64_t count = 5; // ADIOSCountParticles: uint64_t
                ADIOS_SELECTION* piSel = adios_selection_boundingbox(1, &start, &count);

                // avoid deadlock between not finished pmacc tasks and mpi calls in adios
                __getTransactionEvent().waitForFinished();
                ADIOS_CMD(adios_schedule_read(
                    params->fp,
                    piSel,
                    (particlePath + std::string("particles_info")).c_str(),
                    0,
                    1,
                    (void*) particlesInfo));

                /* start a blocking read of all scheduled variables */
                ADIOS_CMD(adios_perform_reads(params->fp, 1));
                adios_selection_delete(piSel);

                /* Run a prefix sum over the numParticles[0] element in particlesInfo
                 * to retreive the offset of particles before gc.getGlobalRank() */
                uint64_t particleOffset = 0;

                uint64_t fullParticlesInfo[gc.getGlobalSize()];

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                __getTransactionEvent().waitForFinished();
                MPI_CHECK(MPI_Allgather(
                    particlesInfo,
                    1,
                    MPI_UINT64_T,
                    fullParticlesInfo,
                    1,
                    MPI_UINT64_T,
                    gc.getCommunicator().getMPIComm()));

                for(size_t i = 0; i < gc.getGlobalSize(); ++i)
                {
                    /* this comparison is potentially harmful, since the order of ranks
                       is not necessarily the same in subsequent MPI jobs.
                       But due to the wrong sorting by rank in `ADIOSCountParticles.hpp`
                       while calculating the `myParticleOffset` we have to immitate that. */
                    if(i < gc.getGlobalRank())
                        particleOffset += fullParticlesInfo[i];
                    if(i == gc.getGlobalRank())
                        totalNumParticles = fullParticlesInfo[i];
                }

                log<picLog::INPUT_OUTPUT>("ADIOS: Loading %1% particles from offset %2%")
                    % (long long unsigned) totalNumParticles % (long long unsigned) particleOffset;

                AdiosFrameType hostFrame;
                log<picLog::INPUT_OUTPUT>("ADIOS: malloc mapped memory: %1%") % speciesName;
                /*malloc mapped memory*/
                meta::ForEach<typename AdiosFrameType::ValueTypeSeq, MallocMemory<bmpl::_1>> mallocMem;
                mallocMem(hostFrame, totalNumParticles);

                log<picLog::INPUT_OUTPUT>("ADIOS: get mapped memory device pointer: %1%") % speciesName;
                /*load device pointer of mapped memory*/
                AdiosFrameType deviceFrame;
                meta::ForEach<typename AdiosFrameType::ValueTypeSeq, GetDevicePtr<bmpl::_1>> getDevicePtr;
                getDevicePtr(deviceFrame, hostFrame);

                meta::ForEach<typename AdiosFrameType::ValueTypeSeq, LoadParticleAttributesFromADIOS<bmpl::_1>>
                    loadAttributes;
                loadAttributes(params, hostFrame, particlePath, particleOffset, totalNumParticles);

                if(totalNumParticles != 0)
                {
                    pmacc::particles::operations::splitIntoListOfFrames(
                        *speciesTmp,
                        deviceFrame,
                        totalNumParticles,
                        restartChunkSize,
                        localDomain.offset,
                        totalCellIdx_,
                        *(params->cellDescription),
                        picLog::INPUT_OUTPUT());

                    /*free host memory*/
                    meta::ForEach<typename AdiosFrameType::ValueTypeSeq, FreeMemory<bmpl::_1>> freeMem;
                    freeMem(hostFrame);
                }
                log<picLog::INPUT_OUTPUT>("ADIOS: ( end ) load species: %1%") % speciesName;
            }
        };


    } /* namespace adios */

} /* namespace picongpu */
