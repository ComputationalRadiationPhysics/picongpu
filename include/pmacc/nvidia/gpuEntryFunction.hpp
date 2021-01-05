/* Copyright 2016-2021 Felix Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once


#include "pmacc/types.hpp"

#pragma once

namespace pmacc
{
    namespace nvidia
    {
        /**
         *
         * @tparam T_KernelFunctor type of the functor for device execution
         */
        template<typename T_KernelFunctor>
        struct PMaccKernel
        {
            /**
             *
             * @param acc functor for device execution
             * @param args arguments for the functor
             */
            template<typename T_Acc, typename... T_Args>
            DINLINE void operator()(T_Acc const acc, T_Args... args) const
            {
                T_KernelFunctor{}(acc, args...);
            }
        };
    } // namespace nvidia
} // namespace pmacc
