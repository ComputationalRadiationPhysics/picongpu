/*
 * SPDX-FileCopyrightText: Rene Widera, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
