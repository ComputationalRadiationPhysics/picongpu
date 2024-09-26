/* Copyright 2013-2023 Rene Widera, Sergei Bastrakov
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
        /** Trait to get margins of a solver type
         *
         * Defines internal types LowerMargin and UpperMargin.
         * Both are compile-time simDim-dimentional integer vectors with the number of margin cells required.
         * Thus, T_Solver applied to any cell gives a guarantee to use not more than LowerMargin cells to the "left"
         * and not more than UpperMargin cells to the "right" from the cell.
         *
         * By default propagates eponimous internal types of T_Solver.
         *
         * @tparam T_Solver solver type
         * @tparam T_Parameter an optional parameter type
         */
        template<typename T_Solver, typename T_Parameter = void>
        struct GetMargin
        {
            using LowerMargin = typename T_Solver::LowerMargin;
            using UpperMargin = typename T_Solver::UpperMargin;
        };

        /** Trait to get a lower margin of a solver type
         *
         * Defines internal type which is a compile-time simDim-dimentional integer vector.
         * T_Solver applied to any cell gives a guarantee to use not more than type cells to the "left" from the cell.
         *
         * By default propagates GetMargin<...>::LowerMargin.
         *
         * @tparam T_Solver solver type
         * @tparam T_Parameter an optional parameter type
         */
        template<typename T_Solver, typename T_Parameter = void>
        struct GetLowerMargin
        {
            using type = typename picongpu::traits::GetMargin<T_Solver, T_Parameter>::LowerMargin;
        };

        /** Trait to get an upper margin of a solver type
         *
         * Defines internal type which is a compile-time simDim-dimentional integer vector.
         * T_Solver applied to any cell gives a guarantee to use not more than type cells to the "right" from the cell.
         *
         * By default propagates GetMargin<...>::UpperMargin.
         *
         * @tparam T_Solver solver type
         * @tparam T_Parameter an optional parameter type
         */
        template<typename T_Solver, typename T_Parameter = void>
        struct GetUpperMargin
        {
            using type = typename picongpu::traits::GetMargin<T_Solver, T_Parameter>::UpperMargin;
        };

    } // namespace traits
} // namespace picongpu
