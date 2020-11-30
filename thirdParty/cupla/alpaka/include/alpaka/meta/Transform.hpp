/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace alpaka
{
    namespace meta
    {
        namespace detail
        {
            //#############################################################################
            template<typename Ts, template<typename...> class TOp>
            struct TransformImpl;
            //#############################################################################
            template<template<typename...> class TList, typename... Ts, template<typename...> class TOp>
            struct TransformImpl<TList<Ts...>, TOp>
            {
                using type = TList<TOp<Ts>...>;
            };
        } // namespace detail
        //#############################################################################
        template<typename Ts, template<typename...> class TOp>
        using Transform = typename detail::TransformImpl<Ts, TOp>::type;
    } // namespace meta
} // namespace alpaka
