/* Copyright 2013-2019 Rene Widera, Felix Schmitt, Axel Huebl, Franz Poeschel
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

#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/openPMD/openPMDWriter.def"
#include "picongpu/plugins/openPMD/restart/LoadParticleAttributesFromOpenPMD.hpp"
#include "picongpu/plugins/output/WriteSpeciesCommon.hpp"
#include "picongpu/simulation_defines.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/meta/conversion/RemoveFromSeq.hpp>
#include <pmacc/particles/ParticleDescription.hpp>
#include <pmacc/particles/operations/splitIntoListOfFrames.kernel>

#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/type_traits.hpp>
#include <boost/type_traits/is_same.hpp>

#include <openPMD/openPMD.hpp>


namespace picongpu
{
    namespace openPMD
    {
        using namespace pmacc;

        /** Load species from openPMD checkpoint storage
         *
         * @tparam T_Species type of species
         */
        template<typename T_Species>
        struct LoadSpecies
        {
        public:
            using ThisSpecies = T_Species;
            using FrameType = typename ThisSpecies::FrameType;
            using ParticleDescription = typename FrameType::ParticleDescription;
            using ParticleAttributeList = typename FrameType::ValueTypeSeq;


            /* delete multiMask and localCellIdx in openPMD particle*/
            using TypesToDelete = bmpl::vector2<multiMask, localCellIdx>;
            using ParticleCleanedAttributeList = typename RemoveFromSeq<ParticleAttributeList, TypesToDelete>::type;

            /* add totalCellIdx for openPMD particle*/
            using ParticleNewAttributeList = typename MakeSeq<ParticleCleanedAttributeList, totalCellIdx>::type;

            using NewParticleDescription =
                typename ReplaceValueTypeSeq<ParticleDescription, ParticleNewAttributeList>::type;

            using openPMDFrameType = Frame<OperatorCreateVectorBox, NewParticleDescription>;

            /** Load species from openPMD checkpoint storage
             *
             * @param params thread params
             * @param restartChunkSize number of particles processed in one kernel
             * call
             */
            HINLINE void operator()(ThreadParams* params, const uint32_t restartChunkSize)
            {
                std::string const speciesName = FrameType::getName();
                log<picLog::INPUT_OUTPUT>("openPMD: (begin) load species: %1%") % speciesName;
                DataConnector& dc = Environment<>::get().DataConnector();
                GridController<simDim>& gc = Environment<simDim>::get().GridController();

                ::openPMD::Series& series = *params->openPMDSeries;
                ::openPMD::Container<::openPMD::ParticleSpecies>& particles
                    = series.iterations[params->currentStep].particles;
                ::openPMD::ParticleSpecies particleSpecies = particles[speciesName];

                const pmacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();

                /* load particle without copying particle data to host */
                auto speciesTmp = dc.get<ThisSpecies>(FrameType::getName(), true);

                /* count total number of particles on the device */
                uint64_t totalNumParticles = 0;

                // avoid deadlock between not finished pmacc tasks and mpi calls in
                // openPMD
                __getTransactionEvent().waitForFinished();
                std::shared_ptr<uint64_t> fullParticlesInfoShared
                    = particleSpecies.particlePatches["numParticles"][::openPMD::RecordComponent::SCALAR]
                          .load<uint64_t>();
                series.flush();

                /* Run a prefix sum over the numParticles[0] element in
                 * particlesInfo to retreive the offset of particles before
                 * gc.getGlobalRank() */
                uint64_t particleOffset = 0;

                uint64_t* fullParticlesInfo = fullParticlesInfoShared.get();

                for(size_t i = 0; i < gc.getGlobalSize(); ++i)
                {
                    /* this comparison is potentially harmful, since the order of
                       ranks is not necessarily the same in subsequent MPI jobs. But
                       due to the wrong sorting by rank in
                       `openPMDCountParticles.hpp` while calculating the
                       `myParticleOffset` we have to immitate that. */
                    if(i < gc.getGlobalRank())
                        particleOffset += fullParticlesInfo[i];
                    if(i == gc.getGlobalRank())
                        totalNumParticles = fullParticlesInfo[i];
                }

                log<picLog::INPUT_OUTPUT>("openPMD: Loading %1% particles from offset %2%")
                    % (long long unsigned) totalNumParticles % (long long unsigned) particleOffset;

                openPMDFrameType hostFrame;
                log<picLog::INPUT_OUTPUT>("openPMD: malloc mapped memory: %1%") % speciesName;
                /*malloc mapped memory*/
                meta::ForEach<typename openPMDFrameType::ValueTypeSeq, MallocMemory<bmpl::_1>> mallocMem;
                mallocMem(hostFrame, totalNumParticles);

                log<picLog::INPUT_OUTPUT>("openPMD: get mapped memory device pointer: %1%") % speciesName;
                /*load device pointer of mapped memory*/
                openPMDFrameType deviceFrame;
                meta::ForEach<typename openPMDFrameType::ValueTypeSeq, GetDevicePtr<bmpl::_1>> getDevicePtr;
                getDevicePtr(deviceFrame, hostFrame);

                meta::ForEach<typename openPMDFrameType::ValueTypeSeq, LoadParticleAttributesFromOpenPMD<bmpl::_1>>
                    loadAttributes;
                loadAttributes(params, hostFrame, particleSpecies, particleOffset, totalNumParticles);

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
                    meta::ForEach<typename openPMDFrameType::ValueTypeSeq, FreeMemory<bmpl::_1>> freeMem;
                    freeMem(hostFrame);
                }
                log<picLog::INPUT_OUTPUT>("openPMD: ( end ) load species: %1%") % speciesName;
            }
        };


    } /* namespace openPMD */

} /* namespace picongpu */
