/* Copyright 2013-2021 Heiko Burau, Rene Widera
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
#include "pmacc/particles/boostExtension/InheritLinearly.hpp"

#include <boost/mpl/map.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/utility/result_of.hpp>


namespace pmacc
{
    namespace math
    {
        namespace bmpl = boost::mpl;

        /** wrap a datum
         *
         * align the data structure with `PMACC_ALIGN`
         *
         * @tparam T_Pair boost mpl pair< key, type of the value >
         */
        template<typename T_Pair>
        struct AlignedData
        {
            typedef typename T_Pair::first Key;
            typedef typename T_Pair::second ValueType;

            PMACC_ALIGN(value, ValueType);

            HDINLINE AlignedData()
            {
            }

            HDINLINE AlignedData(const ValueType& value) : value(value)
            {
            }

            HDINLINE ValueType& operator[](const Key&)
            {
                return value;
            }

            HDINLINE const ValueType& operator[](const Key&) const
            {
                return value;
            }
        };

        /** wrap a datum
         *
         * @tparam T_Pair boost mpl pair< key, type of the value >
         */
        template<typename T_Pair>
        struct NativeData
        {
            typedef typename T_Pair::first Key;
            typedef typename T_Pair::second ValueType;

            ValueType value;

            HDINLINE NativeData()
            {
            }

            HDINLINE NativeData(const ValueType& value) : value(value)
            {
            }

            HDINLINE ValueType& operator[](const Key&)
            {
                return value;
            }

            HDINLINE const ValueType& operator[](const Key&) const
            {
                return value;
            }
        };

        template<typename T_Map, template<typename> class T_PodType = NativeData>
        struct MapTuple : protected InheritLinearly<T_Map, T_PodType>
        {
            typedef T_Map Map;
            static constexpr int dim = bmpl::size<Map>::type::value;
            typedef InheritLinearly<T_Map, T_PodType> Base;

            template<class>
            struct result;

            template<class T_F, class T_Key>
            struct result<T_F(T_Key)>
            {
                typedef typename bmpl::at<Map, T_Key>::type& type;
            };

            template<class T_F, class T_Key>
            struct result<const T_F(T_Key)>
            {
                typedef const typename bmpl::at<Map, T_Key>::type& type;
            };

            /** access a datum with a key
             *
             * @tparam T_Key key type
             *
             * @{
             */
            template<typename T_Key>
            HDINLINE typename boost::result_of<MapTuple(T_Key)>::type operator[](const T_Key& key)
            {
                return (*(static_cast<T_PodType<bmpl::pair<T_Key, typename bmpl::at<Map, T_Key>::type>>*>(this)))[key];
            }

            template<typename T_Key>
            HDINLINE typename boost::result_of<const MapTuple(T_Key)>::type operator[](const T_Key& key) const
            {
                return (*(
                    static_cast<const T_PodType<bmpl::pair<T_Key, typename bmpl::at<Map, T_Key>::type>>*>(this)))[key];
            }
            /** @} */

            /** access a datum with an index
             *
             * @tparam T_i the index of tuple's i-th element
             *
             * @{
             */
            template<int T_i>
            HDINLINE typename boost::result_of<MapTuple(typename bmpl::at<Map, bmpl::int_<T_i>>::type::first)>::type
            at()
            {
                return (*this)[typename bmpl::at<Map, bmpl::int_<T_i>>::type::first()];
            }

            template<int T_i>
            HDINLINE
                typename boost::result_of<const MapTuple(typename bmpl::at<Map, bmpl::int_<T_i>>::type::first)>::type
                at() const
            {
                return (*this)[typename bmpl::at<Map, bmpl::int_<T_i>>::type::first()];
            }
            /** @} */
        };

    } // namespace math
} // namespace pmacc
