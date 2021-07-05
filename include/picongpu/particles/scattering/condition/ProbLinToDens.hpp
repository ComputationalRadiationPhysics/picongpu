/* Copyright 2021 Pawel Ordyna
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

namespace picongpu
{
    namespace particles
    {
        namespace scattering
        {
            namespace condition
            {
                namespace acc
                {
                    template<typename T_ParamClass>
                    struct ProbLinToDens
                    {
                        static constexpr float_X propFactor = T_ParamClass::propFactor;

                        template<typename T_rng, typename T_Particle>
                        DINLINE bool operator()(T_rng& rng, T_Particle const& particle, float_X const& density) const
                        {
                            // the probability is the photon path in on step over the free path length(sigma * n)
                            const float_X probability{
                                propFactor * density / BASE_DENSITY
                                * particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE};
                            return (rng() < probability);
                        }
                    };
                } // namespace acc
            } // namespace condition
        } // namespace scattering
    } // namespace particles
} // namespace picongpu
