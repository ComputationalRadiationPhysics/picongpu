/* Copyright 2017-2023 Axel Huebl
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
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
