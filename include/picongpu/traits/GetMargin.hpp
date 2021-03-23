/* Copyright 2013-2021 Rene Widera
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

namespace picongpu
{
    namespace traits
    {
        /**Get margin of a solver
         * class must define a LowerMargin and UpperMargin for any valid solver
         *
         * \tparam Solver solver which needs ghost cells for solving a problem
         *         if solver not define `LowerMargin` and `UpperMargin` this trait (GetMargin)
         *         must be specialized
         * \tparam T_Parameter an optional parameter type
         * for different objects
         */
        template<class Solver, typename T_Parameter = void>
        struct GetMargin
        {
            using LowerMargin = typename Solver::LowerMargin;
            using UpperMargin = typename Solver::UpperMargin;
        };

        template<typename T_Type>
        struct GetLowerMargin
        {
            typedef typename traits::GetMargin<T_Type>::LowerMargin type;
        };

        template<typename T_Type>
        struct GetUpperMargin
        {
            typedef typename traits::GetMargin<T_Type>::UpperMargin type;
        };

    } // namespace traits

} // namespace picongpu
