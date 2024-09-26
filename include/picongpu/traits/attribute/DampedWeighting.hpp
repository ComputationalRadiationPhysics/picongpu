/* Copyright 2022-2023 Sergei Bastrakov
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

#include "picongpu/defines.hpp"

#include <pmacc/traits/HasIdentifier.hpp>


namespace picongpu
{
    namespace traits
    {
        namespace attribute
        {
            /** Get the damped weighting of a macro particle
             *
             * For particles with weightingDampingFactor attribute it is a product of that factor and weighting.
             * Otherwise, damped weighting is equal to weighting.
             *
             * @tparam T_Particle particle type, must have weighting attribute
             *
             * @param particle a reference to a particle
             * @return damped weighting of the macro particle
             */
            template<typename T_Particle>
            HDINLINE auto getDampedWeighting(const T_Particle& particle)
            {
                using HasWeightingDampingFactor =
                    typename pmacc::traits::HasIdentifier<T_Particle, weightingDampingFactor>::type;
                if constexpr(HasWeightingDampingFactor::value)
                    return particle[weighting_] * particle[weightingDampingFactor_];
                return particle[weighting_];
            }

            /** Damp weighting of a macro particle with a given multiplier
             *
             * For particles with weightingDampingFactor attribute, affects this attribute.
             * Otherwise, affects weighting and weighted momentum to keep physical momentum same.
             *
             * @warning In case there are other weighted attributes they are not changed by this function and
             * so will become inconsistent. For now it has to be handled by a caller, see #4299.
             *
             * @tparam T_Particle particle type, must have weighting attribute
             *
             * @param[in,out] particle a reference to a particle
             * @param multiplier weighting multiplier, values of < 1 result in damping, > 1 result in boosting
             */
            template<typename T_Particle>
            HDINLINE void dampWeighting(T_Particle& particle, float_X minWeighting, float_X const multiplier)
            {
                using HasWeightingDampingFactor =
                    typename pmacc::traits::HasIdentifier<T_Particle, weightingDampingFactor>::type;
                if constexpr(HasWeightingDampingFactor::value)
                    particle[weightingDampingFactor_] *= multiplier;
                else
                {
                    auto const oldWeighting = particle[weighting_];
                    auto const attemptedWeighting = oldWeighting * multiplier;
                    // Cap at MIN_WEIGHTING, copy it to avoid taking address of constexpr in max()
                    auto const dampedWeighting = math::max(attemptedWeighting, minWeighting);
                    particle[weighting_] = dampedWeighting;
                    /* Here we have to update values of all weighted attributes.
                     * We currently have no generic and consistent way to do that #4299.
                     * So at least update weighted momentums so that particles don't artificially accelerate.
                     * This way of updating takes into account the potential capping.
                     */
                    particle[momentum_] *= dampedWeighting / oldWeighting;
                }
            }

        } // namespace attribute
    } // namespace traits
} // namespace picongpu
