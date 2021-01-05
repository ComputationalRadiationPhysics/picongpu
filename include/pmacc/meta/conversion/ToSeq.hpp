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
#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/if.hpp>


namespace pmacc
{
    /** cast type to boost mpl vector
     * @return ::type if T_Type is sequence then identity of T_Type
     *                else boost::mpl::vector<T_Type>
     */
    template<typename T_Type>
    struct ToSeq
    {
        typedef typename bmpl::if_<bmpl::is_sequence<T_Type>, T_Type, bmpl::vector1<T_Type>>::type type;
    };

} // namespace pmacc
