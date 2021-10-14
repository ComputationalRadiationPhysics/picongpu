/* Copyright 2021 Bernhard Manfred Gruber
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
            template<typename List>
            struct Front
            {
            };

            template<template<typename...> class List, typename Head, typename... Tail>
            struct Front<List<Head, Tail...>>
            {
                using type = Head;
            };
        } // namespace detail

        template<typename List>
        using Front = typename detail::Front<List>::type;

        template<typename List, typename Value>
        struct Contains : std::false_type
        {
        };

        template<template<typename...> class List, typename Head, typename... Tail, typename Value>
        struct Contains<List<Head, Tail...>, Value>
        {
            static constexpr bool value = std::is_same<Head, Value>::value || Contains<List<Tail...>, Value>::value;
        };
    } // namespace meta
} // namespace alpaka
