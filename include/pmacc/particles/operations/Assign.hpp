/* Copyright 2013-2021 Rene Widera
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
    namespace particles
    {
        namespace operations
        {
            namespace detail
            {
                template<typename T_Dest, typename T_Src>
                struct Assign;

            } // namespace detail

            template<typename T_Dest, typename T_Src>
            HDINLINE void assign(T_Dest& dest, const T_Src& src)
            {
                detail::Assign<T_Dest, T_Src>()(dest, src);
            }

        } // namespace operations
    } // namespace particles
} // namespace pmacc
