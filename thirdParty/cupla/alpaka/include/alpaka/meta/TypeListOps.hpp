/* Copyright 2022 Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

namespace alpaka::meta
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
        static constexpr bool value = std::is_same_v<Head, Value> || Contains<List<Tail...>, Value>::value;
    };
} // namespace alpaka::meta
