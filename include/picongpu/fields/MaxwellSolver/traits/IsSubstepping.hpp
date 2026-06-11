/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
