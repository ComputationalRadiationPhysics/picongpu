/* Copyright 2026 PIConGPU contributors
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#if (ENABLE_OPENPMD == 1)

#    include "picongpu/defines.hpp"
#    include "picongpu/plugins/openPMD/openPMDWriter.def"
#    include "picongpu/plugins/openPMD/particles/LoadParticleData.hpp"
#    include "picongpu/plugins/openPMD/restart/LoadSpecies.hpp"

#    include <pmacc/dataManagement/DataConnector.hpp>
#    include <pmacc/particles/meta/FindByNameOrType.hpp>

#    include <boost/mpl/apply.hpp>

#    include <cstdint>
#    include <memory>
#    include <string>
#    include <utility>

#    include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace particles
    {
        /** Load a species' particles from an external openPMD file at t=0.
         *
         * Standard-compliant path: each particle's global position is built
         * from ``position`` and ``positionOffset`` (both the constant and the
         * non-constant form are supported, since the unit conversion is applied
         * per component by :class:`LoadParticleAttributesFromOpenPMD`), and
         * ``particlePatches`` are not required. Particles outside the local
         * domain are filtered out. As a fast path, if the file provides
         * ``particlePatches`` in PIConGPU's cell-index layout, the cheaper
         * patch-based restart reader is used instead.
         *
         * Like the other init functors this one must be default-constructible;
         * the compile-time parameters are passed as a parameter class (see
         * ``speciesInitialization.param``), analogous to the density profiles.
         *
         * @tparam T_ParticleSourceParamClass parameter class providing public
         *         static members ``filePath`` (``char const*``), ``iteration``
         *         (integer), ``chunkSize`` (integer) and ``SpeciesType`` (the
         *         PIConGPU species type to load into).
         */
        template<typename T_ParticleSourceParamClass>
        struct LoadFromOpenPMD
        {
            HINLINE void operator()(uint32_t const currentStep)
            {
                using SpeciesType = pmacc::particles::meta::
                    FindByNameOrType_t<VectorAllSpecies, typename T_ParticleSourceParamClass::SpeciesType>;
                using FrameType = typename SpeciesType::FrameType;

                DataConnector& dc = Environment<>::get().DataConnector();
                auto speciesPtr = dc.get<SpeciesType>(FrameType::getName());

                // prepare thread params as the openPMD writer does for a restart
                MPI_Comm communicator = MPI_COMM_NULL;
                MPI_CHECK(MPI_Comm_dup(
                    Environment<simDim>::get().GridController().getCommunicator().getMPIComm(),
                    &communicator));

                openPMD::ThreadParams threadParams;
                threadParams.communicator = communicator;
                threadParams.openPMDSeries = std::make_unique<::openPMD::Series>(
                    T_ParticleSourceParamClass::filePath,
                    ::openPMD::Access::READ_ONLY,
                    communicator,
                    "{}");
                auto cellDescription = speciesPtr->getCellDescription();
                threadParams.cellDescription = &cellDescription;
                threadParams.window = MovingWindow::getInstance().getDomainAsWindow(currentStep);
                threadParams.localWindowToDomainOffset = DataSpace<simDim>::create(0);
                threadParams.isCheckpoint = false;

                std::string const speciesName = FrameType::getName();

                ::openPMD::Series& series = *threadParams.openPMDSeries;
                ::openPMD::ParticleSpecies particleSpecies
                    = series.iterations[T_ParticleSourceParamClass::iteration].open().particles[speciesName];

                SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                DataSpace<simDim> const cellOffsetToTotalDomain
                    = subGrid.getLocalDomain().offset + subGrid.getGlobalDomain().offset;

                if(usesParticlePatches(particleSpecies))
                {
                    // fast path: PIConGPU-layout checkpoint, reuse the restart reader
                    log<picLog::INPUT_OUTPUT>("openPMD: loading species %1% from file via particlePatches fast path")
                        % FrameType::getName();
                    openPMD::LoadSpecies<SpeciesType> loader;
                    loader(
                        &threadParams,
                        T_ParticleSourceParamClass::iteration,
                        T_ParticleSourceParamClass::chunkSize);
                }
                else
                {
                    // general path: treat the whole record as one patch and filter by position
                    log<picLog::INPUT_OUTPUT>(
                        "openPMD: loading species %1% from file as a single patch (standard path)")
                        % speciesName;
                    uint64_t patchNumParticles[1] = {determineNumParticles(particleSpecies)};
                    uint64_t patchNumParticlesOffset[1] = {0};

                    if(patchNumParticles[0] == 0u)
                        throw std::runtime_error(
                            "openPMD: no particles found in species '" + speciesName + "' of the source file.");

                    typename openPMD::ParticleSpeciesLoader<SpeciesType>::Params params{
                        speciesName,
                        particleSpecies,
                        speciesPtr,
                        patchNumParticles,
                        patchNumParticlesOffset,
                        cellOffsetToTotalDomain,
                        T_ParticleSourceParamClass::chunkSize};
                    params.loadPartialMatches({0}, &threadParams);
                }

                particleSpecies.seriesFlush();
                // the series holds the communicator, so destroy it before freeing
                threadParams.openPMDSeries.reset();
                MPI_CHECK(MPI_Comm_free(&communicator));
            }

        private:
            /** Whether the file's particlePatches can be used by the restart reader.
             *
             * Requires numParticles, numParticlesOffset, offset and extent.
             */
            static bool usesParticlePatches(::openPMD::ParticleSpecies& particleSpecies)
            {
                auto& patches = particleSpecies.particlePatches;
                if(!patches.contains("numParticles") || !patches.contains("numParticlesOffset")
                   || !patches.contains("offset") || !patches.contains("extent"))
                    return false;
                try
                {
                    auto const numPatches = patches["numParticles"].getExtent();
                    return !numPatches.empty() && numPatches[0] != 0;
                }
                catch(...)
                {
                    return false;
                }
            }

            /** Number of particles in the species record.
             *
             * Taken from the extent of the first available ``position``
             * component, falling back to a scalar record component.
             */
            static uint64_t determineNumParticles(::openPMD::ParticleSpecies& particleSpecies)
            {
                try
                {
                    auto positionRecord = particleSpecies["position"];
                    for(char const* component : {"x", "y", "z"})
                    {
                        try
                        {
                            auto const extent = positionRecord[component].getExtent();
                            if(!extent.empty())
                                return extent[0];
                        }
                        catch(...)
                        {
                        }
                    }
                }
                catch(...)
                {
                }
                try
                {
                    // scalar fallback (e.g. a 1D record stored without components)
                    auto const extent = particleSpecies["position"].getExtent();
                    if(!extent.empty())
                        return extent[0];
                }
                catch(...)
                {
                }
                return 0u;
            }
        };

    } // namespace particles
} // namespace picongpu

#endif
