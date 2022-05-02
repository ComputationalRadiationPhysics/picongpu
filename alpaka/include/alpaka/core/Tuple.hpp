/* Copyright 2022 Jeffrey Kelling, Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/STLTuple/STLTuple.hpp>

#include <cstddef>
#include <utility>

namespace alpaka::core
{
    using namespace ::utility::tuple;

    namespace detail
    {
        template<typename TFunc, typename... TArgs, std::size_t... Is>
        constexpr auto apply_impl(TFunc&& f, Tuple<TArgs...>&& t, std::index_sequence<Is...>)
        {
            return f(get<Is>(std::forward<Tuple<TArgs...>&&>(t))...);
        }
    } // namespace detail

    template<typename TFunc, typename... TArgs>
    constexpr auto apply(TFunc&& f, Tuple<TArgs...> t)
    {
        return detail::apply_impl(
            std::forward<TFunc>(f),
            std::forward<Tuple<TArgs...>&&>(t),
            std::make_index_sequence<sizeof...(TArgs)>{});
    }
} // namespace alpaka::core
