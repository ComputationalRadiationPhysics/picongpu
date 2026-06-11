/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/meta/Mp11.hpp"
#include "pmacc/traits/HasIdentifier.hpp"

namespace pmacc
{
    namespace traits
    {
        /** Checks if an object has all specified identifiers
         *
         * Individual identifiers checks are logically connected via
         * mp_all_of .
         *
         * @tparam T_Object any object (class or typename)
         * @tparam T_SeqKeys a sequence of identifiers
         *
         * This struct must define
         * ::type (pmacc::mp_bool_<>)
         */
        template<typename T_Object, typename T_SeqKeys>
        struct HasIdentifiers
        {
            template<typename T>
            using Predicate = typename HasIdentifier<T_Object, T>::type;

            using type = pmacc::mp_all_of<T_SeqKeys, Predicate>;
        };

        template<typename T_Object, typename T_SeqKeys>
        bool hasIdentifiers(T_Object const&, T_SeqKeys const&)
        {
            return HasIdentifiers<T_Object, T_SeqKeys>::type::value;
        }

    } // namespace traits
} // namespace pmacc
