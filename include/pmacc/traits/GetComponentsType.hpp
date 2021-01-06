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

#include <boost/type_traits.hpp>

namespace pmacc
{
    namespace traits
    {
        /** Get component type of an object
         *
         * \tparam T_Type any type
         * \return \p ::type get result type
         *            If T_Type is fundamental c++ type, the identity is returned
         *
         * Attention: do not defines this trait for structs with different attributes inside
         */
        template<typename T_Type, bool T_IsFundamental = boost::is_fundamental<T_Type>::value>
        struct GetComponentsType;

        template<typename T_Type>
        struct GetComponentsType<T_Type, true>
        {
            typedef T_Type type;
        };

    } // namespace traits

} // namespace pmacc
