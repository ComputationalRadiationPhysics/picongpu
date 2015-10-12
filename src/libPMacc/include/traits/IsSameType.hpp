/**
 * Copyright 2013 Axel Huebl, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

namespace PMacc
{
    /** IsSameType< typeA, typeB >
     *
     *  Compare two types, IsSameType::result is true if they are equal,
     *  else false.
     *
     *  \tparam T1 first type to compare
     *  \tparam T2 second type to compare
     */
    template< typename T1, typename T2 >
    struct IsSameType
    {
        BOOST_STATIC_CONSTEXPR bool result = false;
    };

    template< typename T1 >
    struct IsSameType< T1, T1 >
    {
        BOOST_STATIC_CONSTEXPR bool result = true;
    };

} // namespace PMacc
