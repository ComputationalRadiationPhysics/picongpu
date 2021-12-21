/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <type_traits>

namespace alpaka
{
    namespace meta
    {
        namespace detail
        {
            //#############################################################################
            // TODO: Replace with C++17 std::conjunction
            template<typename...>
            struct ConjunctionImpl : std::true_type
            {
            };
            //#############################################################################
            // TODO: Replace with C++17 std::conjunction
            template<typename B1>
            struct ConjunctionImpl<B1> : B1
            {
            };
            //#############################################################################
            // TODO: Replace with C++17 std::conjunction
            template<typename B1, typename... Bn>
            struct ConjunctionImpl<B1, Bn...> : std::conditional<B1::value != false, ConjunctionImpl<Bn...>, B1>::type
            {
            };
        } // namespace detail
        //#############################################################################
        template<typename... B>
        using Conjunction = typename detail::ConjunctionImpl<B...>::type;

        namespace detail
        {
            //#############################################################################
            // TODO: Replace with C++17 std::disjunction
            template<typename...>
            struct DisjunctionImpl : std::false_type
            {
            };
            //#############################################################################
            // TODO: Replace with C++17 std::disjunction
            template<typename B1>
            struct DisjunctionImpl<B1> : B1
            {
            };
            //#############################################################################
            // TODO: Replace with C++17 std::disjunction
            template<typename B1, typename... Bn>
            struct DisjunctionImpl<B1, Bn...> : std::conditional<B1::value != false, B1, DisjunctionImpl<Bn...>>::type
            {
            };
        } // namespace detail
        //#############################################################################
        template<typename... B>
        using Disjunction = typename detail::DisjunctionImpl<B...>;

        //#############################################################################
        // TODO: Replace with C++17 std::negation
        template<typename B>
        using Negation = std::integral_constant<bool, !B::value>;
    } // namespace meta
} // namespace alpaka
