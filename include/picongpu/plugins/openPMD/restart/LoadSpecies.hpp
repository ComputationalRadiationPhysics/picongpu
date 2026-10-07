/* Copyright 2013-2024 Rene Widera, Felix Schmitt, Axel Huebl, Franz Poeschel
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
#    include "picongpu/plugins/output/WriteSpeciesCommon.hpp"

#    include <boost/mpl/placeholders.hpp>

#    include <deque>
#    include <utility>

#    include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace openPMD
    {
        using namespace pmacc;

        /** Load species from openPMD checkpoint storage
         *
         * The particle-attribute reading, in-domain filtering and scatter are
         * shared with the general from-file loader via
         * :class:`picongpu::openPMD::ParticleSpeciesLoader`; this class only adds
         * the restart-specific patch matching, which uses the file's
         * ``particlePatches`` (given in total-domain cell indices) to find the
         * patches belonging to the local MPI domain.
         *
         * @tparam T_Species type of species
         */
        template<typename T_Species>
        struct LoadSpecies : public ParticleSpeciesLoader<T_Species>
        {
            using Base = ParticleSpeciesLoader<T_Species>;
            using ThisSpecies = typename Base::ThisSpecies;
            using FrameType = typename Base::FrameType;
            using LoadParams = typename Base::Params;

            /** Load species from openPMD checkpoint storage
             *
             * @param params thread params
             * @param restartChunkSize number of particles processed in one kernel
             * call
             */
            HINLINE void operator()(ThreadParams* params, uint32_t const currentStep, uint32_t const restartChunkSize)
            {
                std::string const speciesName = FrameType::getName();

                // avoid deadlock between not finished pmacc tasks and mpi calls in
                // openPMD
                eventSystem::getTransactionEvent().waitForFinished();

                ::openPMD::Series& series = *params->openPMDSeries;
                ::openPMD::Container<::openPMD::ParticleSpecies>& particles
                    = series.iterations[currentStep].open().particles;
                ::openPMD::ParticleSpecies particleSpecies = particles[speciesName];

                SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                DataSpace<simDim> cellOffsetToTotalDomain
                    = subGrid.getLocalDomain().offset + subGrid.getGlobalDomain().offset;

                /* load particle without copying particle data to host */
                DataConnector& dc = Environment<>::get().DataConnector();
                auto speciesTmp = dc.get<ThisSpecies>(FrameType::getName());

                /* find the openPMD patches matching the local domain */
                auto [fullMatches, partialMatches] = getPatchIdx(params, particleSpecies);

                std::shared_ptr<uint64_t> numParticlesShared
                    = particleSpecies.particlePatches["numParticles"].load<uint64_t>();
                std::shared_ptr<uint64_t> numParticlesOffsetShared
                    = particleSpecies.particlePatches["numParticlesOffset"].load<uint64_t>();
                particles.seriesFlush();
                uint64_t* patchNumParticles = numParticlesShared.get();
                uint64_t* patchNumParticlesOffset = numParticlesOffsetShared.get();

                LoadParams lp{
                    speciesName,
                    particleSpecies,
                    speciesTmp,
                    patchNumParticles,
                    patchNumParticlesOffset,
                    cellOffsetToTotalDomain,
                    restartChunkSize};
                lp.loadFullMatches(fullMatches, params);
                lp.loadPartialMatches(partialMatches, params);

                log<picLog::INPUT_OUTPUT>("openPMD: ( end ) load species: %1%") % speciesName;
            }

        private:
            // o: offset, e: extent, u: upper corner (= o+e)
            static std::pair<DataSpace<simDim>, DataSpace<simDim>> intersect(
                DataSpace<simDim> const& o1,
                DataSpace<simDim> const& e1,
                DataSpace<simDim> const& o2,
                DataSpace<simDim> const& e2)
            {
                // Convert extents into upper coordinates
                auto u1 = o1 + e1;
                auto u2 = o2 + e2;

                DataSpace<simDim> intersect_o, intersect_u, intersect_e;
                for(unsigned d = 0; d < simDim; ++d)
                {
                    intersect_o[d] = std::max(o1[d], o2[d]);
                    intersect_u[d] = std::min(u1[d], u2[d]);
                    intersect_e[d] = intersect_u[d] > intersect_o[d] ? intersect_u[d] - intersect_o[d] : 0;
                }
                return {intersect_o, intersect_e};
            }

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
            HINLINE std::pair<std::deque<size_t>, std::deque<size_t>> getPatchIdx(
                ThreadParams* params,
                ::openPMD::ParticleSpecies particleSpecies)
            {
                std::string const name_lookup[] = {"x", "y", "z"};

                size_t patches = particleSpecies.particlePatches["numParticles"].getExtent()[0];

                std::vector<DataSpace<simDim>> offsets(patches);
                std::vector<DataSpace<simDim>> extents(patches);

                // transform openPMD particle patch data into PIConGPU data objects
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    std::shared_ptr<uint64_t> patchOffsetsInfoShared
                        = particleSpecies.particlePatches["offset"][name_lookup[d]].load<uint64_t>();
                    std::shared_ptr<uint64_t> patchExtentsInfoShared
                        = particleSpecies.particlePatches["extent"][name_lookup[d]].load<uint64_t>();
                    particleSpecies.seriesFlush();
                    for(size_t i = 0; i < patches; ++i)
                    {
                        offsets[i][d] = patchOffsetsInfoShared.get()[i];
                        extents[i][d] = patchExtentsInfoShared.get()[i];
                    }
                }

                SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                pmacc::Selection<simDim> const localDomain = subGrid.getLocalDomain();
                pmacc::Selection<simDim> const globalDomain = subGrid.getGlobalDomain();
                /* Offset to transform local particle offsets into total offsets for all particles within the
                 * current local domain.
                 * @attention A window can be the full simulation domain or the moving window.
                 */
                DataSpace<simDim> localToTotalDomainOffset(localDomain.offset + globalDomain.offset);

                /* params->localWindowToDomainOffset is in PIConGPU for a restart zero but to stay generic we take
                 * the variable into account.
                 */
                DataSpace<simDim> const patchTotalOffset
                    = localToTotalDomainOffset + params->localWindowToDomainOffset;
                DataSpace<simDim> const patchExtent = params->window.localDimensions.size;
                math::Vector<bool, simDim> true_;
                for(unsigned d = 0; d < simDim; ++d)
                {
                    true_[d] = true;
                }

                // search the patch index based on the offset and extents of local domain size
                std::deque<size_t> fullMatches;
                std::deque<size_t> partialMatches;
                size_t noMatches = 0;
                for(size_t i = 0; i < patches; ++i)
                {
                    if((patchTotalOffset <= offsets[i]) == true_
                       && ((offsets[i] + extents[i]) <= (patchTotalOffset + patchExtent)) == true_)
                    {
                        fullMatches.emplace_back(i);
                    }
                    else if(
                        intersect(offsets[i], extents[i], patchTotalOffset, patchExtent).second.productOfComponents()
                        != 0)
                    {
                        partialMatches.emplace_back(i);
                    }
                    else
                    {
                        ++noMatches;
                    }
                }

                log<picLog::INPUT_OUTPUT>(
                    "openPMD: Found %1% fully and %2% partially matching particle patch(es). %3% "
                    "patch was / patches were not matched.")
                    % fullMatches.size() % partialMatches.size() % noMatches;

                return std::make_pair(std::move(fullMatches), std::move(partialMatches));
            }
        };


    } /* namespace openPMD */

} /* namespace picongpu */

#endif
