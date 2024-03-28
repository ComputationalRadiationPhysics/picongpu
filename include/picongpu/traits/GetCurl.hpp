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
