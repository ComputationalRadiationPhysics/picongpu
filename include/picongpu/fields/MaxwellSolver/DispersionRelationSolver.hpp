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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/MaxwellSolver/DispersionRelation.hpp"


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Numerical solver for dispersion relation for the given field solver
             *
             * Relies on dispersion properties given by DispersionRelation<T_FieldSolver>.
             *
             * @tparam T_FieldSolver field solver type, must have DispersionRelation<> properly specialized
             */
            template<typename T_FieldSolver>
            struct DispersionRelationSolver
            {
                /** Numerically solve the dispersion relation with the given parameters
                 *
                 * @param omega angular frequency = 2pi * c / lambda
                 * @param direction normalized propagation direction
                 * @return absolute value of (angular) wave vector
                 */
                float_64 operator()(float_64 omega, float3_64 direction) const
                {
                    auto dispersion = DispersionRelation<T_FieldSolver>{omega, direction};
                    /* Use simple Newton's method to numerically solve equation dispersion.relation(kAbs) = 0.
                     * We use kAbs corresponding to no dispersion (so matching physical vacuum) as initial
                     * approximation. Since realistic setups must be quite close to it, we should converge to a
                     * solution very quickly. So we are not particularly picky wrt numerical solver parameters.
                     */
                    auto kAbs = omega * SPEED_OF_LIGHT;
                    auto const maxNumSteps = 100;
                    for(uint32_t d = 0; d < maxNumSteps; d++)
                    {
                        auto const derivative = dispersion.relationDerivative(kAbs);
                        // Stop when we are nearly at a stationary point
                        constexpr auto derivativeMargin = 1e-5;
                        if(math::abs(derivative) < derivativeMargin)
                            break;
                        kAbs = kAbs - dispersion.relation(kAbs) / derivative;
                    }
                    return kAbs;
                }
            };

        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu
