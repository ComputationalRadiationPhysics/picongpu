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

#include "picongpu/fields/MaxwellSolver/Substepping/Substepping.def"

#include <type_traits>

namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            namespace traits
            {
                /** Trait to check if the given field solver is substepping
                 *
                 * The default implementation is not-substepping.
                 *
                 * @tparam T_FieldSolver field solver
                 */
                template<typename T_FieldSolver>
                struct IsSubstepping : std::false_type
                {
                };

                /** Specialization for substepping field solvers
                 *
                 * @tparam T_BaseSolver base field solver, follows requirements of field solvers
                 * @tparam T_numSubsteps number of substeps per PIC time iteration
                 */
                template<typename T_BaseSolver, uint32_t T_numSubsteps>
                struct IsSubstepping<Substepping<T_BaseSolver, T_numSubsteps>> : std::true_type
                {
                };
            } // namespace traits
        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu
