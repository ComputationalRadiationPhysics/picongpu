/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/particles/boostExtension/InheritLinearly.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace math
    {
        namespace bmpl = boost::mpl;

        /** wrap a datum
         *
         * @tparam T_Pair mp_list< key, type of the value >
         */
        template<typename T_Pair>
        struct TaggedValue
        {
            using Key = mp_first<T_Pair>;
            using ValueType = mp_second<T_Pair>;

            ValueType value;
        };

        template<typename T_Map>
        struct MapTuple : protected InheritLinearly<T_Map, TaggedValue>
        {
            template<typename T_Key>
            using TaggedValueFor = TaggedValue<mp_map_find<T_Map, T_Key>>;

            /** access a value with a key
             *
             * @tparam T_Key key type
             *
             * @{
             */
            template<typename T_Key>
            HDINLINE auto& operator[](T_Key const& key)
            {
                return static_cast<TaggedValueFor<T_Key>&>(*this).value;
            }

            template<typename T_Key>
            HDINLINE const auto& operator[](T_Key const& key) const
            {
                return static_cast<TaggedValueFor<T_Key>&>(*this).value;
            }

            /** @} */
        };

    } // namespace math
} // namespace pmacc
