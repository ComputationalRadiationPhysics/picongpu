/* Copyright 2021-2023 Sergei Bastrakov
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

#include "picongpu/fields/MaxwellSolver/Substepping/Substepping.def"


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Functor to compile-time get time step used inside the given field solver
             *
             * The default implementation uses same time step as in general PIC.
             *
             * @tparam T_FieldSolver field solver typedef
             */
            template<typename T_FieldSolver>
            struct GetTimeStep
            {
                //! Get the time step value
                HDINLINE constexpr float_X operator()()
                {
                    return DELTA_T;
                }
            };

            /** Specialization of functor to compile-time get time step used inside a substepping field solver
             *
             * @tparam T_BaseSolver base field solver, follows requirements of field solvers
             * @tparam T_numSubsteps number of substeps per PIC time iteration
             */
            template<typename T_BaseSolver, uint32_t T_numSubsteps>
            struct GetTimeStep<Substepping<T_BaseSolver, T_numSubsteps>>
            {
                //! Get the time step value
                HDINLINE constexpr float_X operator()()
                {
                    return DELTA_T / static_cast<float_X>(T_numSubsteps);
                }
            };

            //! Get time step used inside the field solver
            HDINLINE constexpr float_X getTimeStep()
            {
                return GetTimeStep<Solver>{}();
            }

        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu
