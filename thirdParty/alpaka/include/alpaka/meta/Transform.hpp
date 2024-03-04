/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::meta
{
    namespace detail
    {
        template<typename Ts, template<typename...> class TOp>
        struct TransformImpl;

        template<template<typename...> class TList, typename... Ts, template<typename...> class TOp>
        struct TransformImpl<TList<Ts...>, TOp>
        {
            using type = TList<TOp<Ts>...>;
        };
    } // namespace detail
    template<typename Ts, template<typename...> class TOp>
    using Transform = typename detail::TransformImpl<Ts, TOp>::type;
} // namespace alpaka::meta
