/* Copyright 2024 Rene Widera
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

#include "picongpu/simulation/stage/ParticleInit.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/param/particleFilters.param"
#include "picongpu/param/speciesInitialization.param"
#include "picongpu/particles/boundary/RemoveOuterParticles.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/manipulators/manipulators.hpp"

#include <pmacc/functor/Call.hpp>
#include <pmacc/meta/ForEach.hpp>

#include <cstdint>

namespace picongpu::simulation::stage
{
    namespace particles
    {
        /** Remove all particles of the species that are outside the respective boundaries
         *
         * Must be called only for species with a pusher
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of particle species that is checked
         */
        template<typename T_SpeciesType>
        struct RemoveOuterParticles
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            HINLINE void operator()(const uint32_t currentStep) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName());
                picongpu::particles::boundary::removeOuterParticles(*species, currentStep);
            }
        };

        //! Remove all particles of all species with pusher flag that are outside the respective boundaries
        struct RemoveOuterParticlesAllSpecies
        {
            /** Remove all external particles
             *
             * @param currentStep current simulation step
             */
            HINLINE void operator()(const uint32_t currentStep) const
            {
                using VectorSpeciesWithPusher =
                    typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, particlePusher<>>::type;
                meta::ForEach<VectorSpeciesWithPusher, RemoveOuterParticles<boost::mpl::_1>> removeOuterParticles;
                removeOuterParticles(currentStep);
            }
        };
    } // namespace particles

    //! Initialize particles
    void ParticleInit::operator()(uint32_t const step) const
    {
        meta::ForEach<picongpu::particles::InitPipeline, pmacc::functor::Call<boost::mpl::_1>> initSpecies;
        initSpecies(step);
        /* Remove all particles that are outside the respective boundaries
         * (this can happen if density functor didn't account for it).
         * For the rest of the simulation we can be sure the only external particles just crossed the
         * border.
         */
        particles::RemoveOuterParticlesAllSpecies removeOuterParticlesAllSpecies;
        removeOuterParticlesAllSpecies(step);
    }

} // namespace picongpu::simulation::stage
