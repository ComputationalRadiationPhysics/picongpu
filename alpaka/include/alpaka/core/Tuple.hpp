/* Copyright 2022 Jeffrey Kelling, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/STLTuple/STLTuple.hpp"

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
