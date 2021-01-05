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
#include <boost/mpl/bool.hpp>
#include <boost/mpl/vector.hpp>
#include "pmacc/meta/conversion/ToSeq.hpp"
#include <boost/utility/result_of.hpp>

namespace pmacc
{
    namespace particles
    {
        namespace operations
        {
            namespace detail
            {
                /* functor for deselect attributes of an object
                 *
                 * - must be boost result_of compatible
                 * - must define a operator()(T_Object)
                 *
                 * @tparam T_Sequence any boost mpl sequence
                 * @tparam T_Object a type were we can deselect attributes from
                 */
                template<typename T_Sequence, typename T_Object>
                struct Deselect;

            } // namespace detail

            template<typename T_Exclude, typename T_Object>
            HDINLINE
                typename boost::result_of<detail::Deselect<typename ToSeq<T_Exclude>::type, T_Object>(T_Object)>::type
                deselect(T_Object& object)
            {
                typedef typename ToSeq<T_Exclude>::type DeselectSeq;
                typedef detail::Deselect<DeselectSeq, T_Object> BaseType;

                return BaseType()(object);
            }

        } // namespace operations
    } // namespace particles
} // namespace pmacc
