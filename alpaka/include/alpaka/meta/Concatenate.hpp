/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
