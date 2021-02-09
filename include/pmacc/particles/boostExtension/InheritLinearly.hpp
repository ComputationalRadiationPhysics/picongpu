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


#include "pmacc/meta/accessors/Identity.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/inherit.hpp>
#include <boost/mpl/inherit_linearly.hpp>
#include <boost/mpl/placeholders.hpp>

namespace pmacc
{
    namespace detail
    {
        /** get combined type which inherit from a boost mpl sequence
         *
         * @tparam T_Sequence boost mpl sequence with classes
         * @tparam T_Accessor unary operator to transform each element of the sequence
         */
        template<typename T_Sequence, template<typename> class T_Accessor = meta::accessors::Identity>
        using InheritLinearly =
            typename bmpl::inherit_linearly<T_Sequence, bmpl::inherit<bmpl::_1, T_Accessor<bmpl::_2>>>::type;

    } // namespace detail

    /** type which inherits from multiple classes
     *
     * @tparam T_Sequence boost mpl sequence with classes
     * @tparam T_Accessor unary operator to transform each element of the sequence
     */
    template<typename T_Sequence, template<typename> class T_Accessor = meta::accessors::Identity>
    struct InheritLinearly : detail::InheritLinearly<T_Sequence, T_Accessor>
    {
    };

} // namespace pmacc
