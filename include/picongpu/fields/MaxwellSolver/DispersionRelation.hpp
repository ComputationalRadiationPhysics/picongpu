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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/MaxwellSolver/GetTimeStep.hpp"

#include <cstdint>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            //! Helper class that dispersion relation implementations may (but do not have to) inherit
            class DispersionRelationBase
            {
            public:
                /** Create an instance with the given parameters
                 *
                 * @param omega angular frequency = 2pi * c / lambda
                 * @param direction normalized propagation direction
                 */
                DispersionRelationBase(float_64 const omega, float3_64 const direction)
                    : omega(omega)
                    , direction(direction)
                {
                }

            protected:
                //! Field solver time step as float_64
                float_64 const timeStep = static_cast<float_64>(getTimeStep());

                //! Grid steps as float_64
                float3_64 const step = precisionCast<float_64>(cellSize);

                //! Angular frequency
                float_64 const omega;

                //! Normalized propagation direction
                float3_64 const direction;
            };

            /** General implementation follows the physical dispersion relation in vacuum
             *
             * No finite-difference numerical solver exactly adheres to the physical relation in general case.
             * This implementation is given as an ideal-case example and to better illustrate the trait.
             */
            template<typename T_FieldSolver>
            class DispersionRelation : public DispersionRelationBase
            {
            public:
                /** Create a functor with the given parameters
                 *
                 * @param omega (angular) frequency = 2pi * c / lambda
                 * @param direction normalized propagation direction
                 */
                DispersionRelation(float_64 const omega, float3_64 const direction)
                    : DispersionRelationBase(omega, direction)
                {
                }

                /** Calculate f(absK) in the dispersion relation
                 *
                 * @param absK absolute value of the (angular) wave number
                 */
                float_64 relation(float_64 const absK) const
                {
                    // (4.13) in Taflove-Hagness, expressed as rhs - lhs = 0
                    auto rhs = 0.0;
                    for(uint32_t d = 0; d < simDim; d++)
                    {
                        auto const term = absK * direction[d];
                        rhs += term * term;
                    }
                    auto const lhsTerm = omega / SPEED_OF_LIGHT;
                    auto const lhs = lhsTerm * lhsTerm;
                    return rhs - lhs;
                }

                /** Calculate df(absK)/d(absK) in the dispersion relation
                 *
                 * @param absK absolute value of the (angular) wave number
                 */
                float_64 relationDerivative(float_64 const absK) const
                {
                    auto result = 0.0;
                    for(uint32_t d = 0; d < simDim; d++)
                    {
                        // Calculate d(term^2(absK))/d(absK), where term is from relation()
                        auto const term = absK * direction[d];
                        auto const termDerivative = direction[d];
                        result += 2.0 * term * termDerivative;
                    }
                    return result;
                }
            };

        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu
