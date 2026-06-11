/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
         * ::type (pmacc::mp_bool_<>)
         */
        template<typename T_Object, typename T_Key>
        struct HasFlag;

        template<typename T_Object, typename T_Key>
        bool hasFlag(T_Object const& obj, T_Key const& key)
        {
            return HasFlag<T_Object, T_Key>::type::value;
        }

    } // namespace traits

} // namespace pmacc
