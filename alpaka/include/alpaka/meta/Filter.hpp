/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/meta/Concatenate.hpp>

#include <type_traits>

namespace alpaka::meta
{
    namespace detail
    {
        template<template<typename...> class TList, template<typename> class TPred, typename... Ts>
        struct FilterImplHelper;
        template<template<typename...> class TList, template<typename> class TPred>
        struct FilterImplHelper<TList, TPred>
        {
            using type = TList<>;
        };
        template<template<typename...> class TList, template<typename> class TPred, typename T, typename... Ts>
        struct FilterImplHelper<TList, TPred, T, Ts...>
        {
            using type = std::conditional_t<
                TPred<T>::value,
                Concatenate<TList<T>, typename FilterImplHelper<TList, TPred, Ts...>::type>,
                typename FilterImplHelper<TList, TPred, Ts...>::type>;
        };

        template<typename TList, template<typename> class TPred>
        struct FilterImpl;
        template<template<typename...> class TList, template<typename> class TPred, typename... Ts>
        struct FilterImpl<TList<Ts...>, TPred>
        {
            using type = typename detail::FilterImplHelper<TList, TPred, Ts...>::type;
        };
    } // namespace detail
    template<typename TList, template<typename> class TPred>
    using Filter = typename detail::FilterImpl<TList, TPred>::type;
} // namespace alpaka::meta
