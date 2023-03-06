/* Copyright 2013-2022 Heiko Burau, Rene Widera
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

#include "pmacc/particles/boostExtension/InheritLinearly.hpp"
#include "pmacc/types.hpp"

#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/pair.hpp>


namespace pmacc
{
    namespace math
    {
        namespace bmpl = boost::mpl;

        /** wrap a datum
         *
         * @tparam T_Pair boost mpl pair< key, type of the value >
         */
        template<typename T_Pair>
        struct TaggedValue
        {
            using Key = typename T_Pair::first;
            using ValueType = typename T_Pair::second;

            ValueType value;
        };

        template<typename T_Map>
        struct MapTuple : protected InheritLinearly<T_Map, TaggedValue>
        {
            template<typename T_Key>
            using TaggedValueFor = TaggedValue<bmpl::pair<T_Key, typename bmpl::at<T_Map, T_Key>::type>>;

            /** access a value with a key
             *
             * @tparam T_Key key type
             *
             * @{
             */
            template<typename T_Key>
            HDINLINE auto& operator[](const T_Key& key)
            {
                return static_cast<TaggedValueFor<T_Key>&>(*this).value;
            }

            template<typename T_Key>
            HDINLINE const auto& operator[](const T_Key& key) const
            {
                return static_cast<TaggedValueFor<T_Key>&>(*this).value;
            }
            /** @} */
        };

    } // namespace math
} // namespace pmacc
