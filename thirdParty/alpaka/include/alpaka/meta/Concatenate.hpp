/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::meta
{
    namespace detail
    {
        template<typename... T>
        struct ConcatenateImpl;

        template<typename T>
        struct ConcatenateImpl<T>
        {
            using type = T;
        };

        template<template<typename...> class TList, typename... As, typename... Bs, typename... TRest>
        struct ConcatenateImpl<TList<As...>, TList<Bs...>, TRest...>
        {
            using type = typename ConcatenateImpl<TList<As..., Bs...>, TRest...>::type;
        };
    } // namespace detail

    template<typename... T>
    using Concatenate = typename detail::ConcatenateImpl<T...>::type;
} // namespace alpaka::meta
