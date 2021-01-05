/* Copyright 2017-2021 Heiko Burau
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


namespace pmacc
{
    namespace algorithm
    {
        namespace functor
        {
            struct Add
            {
                template<typename T_Type>
                HDINLINE T_Type operator()(T_Type const& first, T_Type const& second) const
                {
                    return first + second;
                }

                template<typename T_Type, typename T_Acc>
                HDINLINE T_Type operator()(T_Acc const&, T_Type const& first, T_Type const& second) const
                {
                    return first + second;
                }
            };

        } // namespace functor
    } // namespace algorithm
} // namespace pmacc
