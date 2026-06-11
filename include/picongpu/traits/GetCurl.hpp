/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

namespace picongpu
{
    namespace traits
    {
        /** Trait for curl(E) type of a field solver
         *
         * Defines the resulting type as ::type.
         * Does not fall back to T_FieldSolver::CurlE by default to prevent circular dependencies.
         * (These dependencies cause compile errors as they cause use of incomplete types.)
         *
         * @tparam T_FieldSolver field solver type
         */
        template<typename T_FieldSolver>
        struct GetCurlE;

        /** Trait for curl(B) type of a field solver
         *
         * Defines the resulting type as ::type.
         * Does not fall back to T_FieldSolver::CurlE by default to prevent circular dependencies.
         * (These dependencies cause compile errors as they cause use of incomplete types.)
         *
         * @tparam T_FieldSolver field solver type
         */
        template<typename T_FieldSolver>
        struct GetCurlB;

    } // namespace traits
} // namespace picongpu
