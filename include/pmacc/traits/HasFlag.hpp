/* Copyright 2014-2021 Rene Widera
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

namespace pmacc
{
    namespace traits
    {
        /** Checks if a Objects has an flag
         *
         * @tparam T_Object any object (class or typename)
         * @tparam T_Key a class which is used as identifier
         *
         * This struct must define
         * ::type (boost::mpl::bool_<>)
         */
        template<typename T_Object, typename T_Key>
        struct HasFlag;

        template<typename T_Object, typename T_Key>
        bool hasFlag(const T_Object& obj, const T_Key& key)
        {
            return HasFlag<T_Object, T_Key>::type::value;
        }

    } // namespace traits

} // namespace pmacc
