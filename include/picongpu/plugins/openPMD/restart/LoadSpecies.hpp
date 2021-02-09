/* Copyright 2013-2021 Rene Widera, Felix Schmitt, Axel Huebl, Franz Poeschel
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

#include <cassert>

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

                const pmacc::Selection<simDim> localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();

                /* load particle without copying particle data to host */
                auto speciesTmp = dc.get<ThisSpecies>(FrameType::getName(), true);

                // avoid deadlock between not finished pmacc tasks and mpi calls in
                // openPMD
                __getTransactionEvent().waitForFinished();

                auto numRanks = gc.getGlobalSize();

                size_t patchIdx = getPatchIdx(params, series, particleSpecies, numRanks);

                std::shared_ptr<uint64_t> fullParticlesInfoShared
                    = particleSpecies.particlePatches["numParticles"][::openPMD::RecordComponent::SCALAR]
                          .load<uint64_t>();
                series.flush();
                uint64_t* fullParticlesInfo = fullParticlesInfoShared.get();

                /* Run a prefix sum over the numParticles[0] element in
                 * particlesInfo to retreive the offset of particles
                 */
                uint64_t particleOffset = 0u;
                /* count total number of particles on the device */
                uint64_t totalNumParticles = 0u;

                assert(patchIdx < numRanks);

                for(size_t i = 0u; i <= patchIdx; ++i)
                {
                    if(i < patchIdx)
                        particleOffset += fullParticlesInfo[i];
                    if(i == patchIdx)
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

        private:
            /** get index for particle data within the openPMD patch data
             *
             * It is not possible to assume that we can use the MPI rank to load the particle data.
             * There is no guarantee that the MPI rank is corresponding to the position within
             * the simulation volume.
             *
             * Use patch information offset and extent to find the index which should be used
             * to load openPMD particle patch data.
             *
             * @return index of the particle patch within the openPMD data
             */
            HINLINE size_t getPatchIdx(
                ThreadParams* params,
                ::openPMD::Series& series,
                ::openPMD::ParticleSpecies particleSpecies,
                size_t numRanks)
            {
                const std::string name_lookup[] = {"x", "y", "z"};

                std::vector<DataSpace<simDim>> offsets(numRanks);
                std::vector<DataSpace<simDim>> extents(numRanks);

                // transform openPMD particle patch data into PIConGPU data objects
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    std::shared_ptr<uint64_t> patchOffsetsInfoShared
                        = particleSpecies.particlePatches["offset"][name_lookup[d]].load<uint64_t>();
                    std::shared_ptr<uint64_t> patchExtentsInfoShared
                        = particleSpecies.particlePatches["extent"][name_lookup[d]].load<uint64_t>();
                    series.flush();
                    for(size_t i = 0; i < numRanks; ++i)
                    {
                        offsets[i][d] = patchOffsetsInfoShared.get()[i];
                        extents[i][d] = patchExtentsInfoShared.get()[i];
                    }
                }

                pmacc::Selection<simDim> const globalDomain = Environment<simDim>::get().SubGrid().getGlobalDomain();
                DataSpace<simDim> const patchOffset = globalDomain.offset + params->window.globalDimensions.offset
                    + params->window.localDimensions.offset;
                DataSpace<simDim> const patchExtent = params->window.localDimensions.size;

                size_t patchIdx = 0;
                // search the patch index based on the offset and extents of local domain size
                for(size_t i = 0; i < numRanks; ++i)
                {
                    if(patchOffset == offsets[i] && patchExtent == extents[i])
                    {
                        patchIdx = i;
                        break;
                    }
                }
                return patchIdx;
            }
        };


    } /* namespace openPMD */

} /* namespace picongpu */
