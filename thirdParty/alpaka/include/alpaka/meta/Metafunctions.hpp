/**
 * \file
 * Copyright 2015 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
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
            template<
                typename...>
            struct ConjunctionImpl :
                std::true_type
            {};
            //#############################################################################
            // TODO: Replace with C++17 std::conjunction
            template<
                typename B1>
            struct ConjunctionImpl<B1> :
                B1
            {};
            //#############################################################################
            // TODO: Replace with C++17 std::conjunction
            template<
                typename B1,
                typename... Bn>
            struct ConjunctionImpl<
                B1,
                Bn...> :
                    std::conditional<B1::value != false, ConjunctionImpl<Bn...>, B1>::type
            {};
        }
        //#############################################################################
        template<
            typename... B>
        using Conjunction = typename detail::ConjunctionImpl<B...>::type;

        namespace detail
        {
            //#############################################################################
            // TODO: Replace with C++17 std::disjunction
            template<
                typename...>
            struct DisjunctionImpl :
                std::false_type
            {};
            //#############################################################################
            // TODO: Replace with C++17 std::disjunction
            template<
                typename B1>
            struct DisjunctionImpl<B1> :
                B1
            {};
            //#############################################################################
            // TODO: Replace with C++17 std::disjunction
            template<
                typename B1,
                typename... Bn>
            struct DisjunctionImpl<
                B1,
                Bn...> :
                    std::conditional<B1::value != false, B1, DisjunctionImpl<Bn...>>::type
            {};
        }
        //#############################################################################
        template<
            typename... B>
        using Disjunction = typename detail::DisjunctionImpl<B...>;

        //#############################################################################
        // TODO: Replace with C++17 std::negation
        template<
            typename B>
        using Negation = std::integral_constant<bool, !B::value>;
    }
}

