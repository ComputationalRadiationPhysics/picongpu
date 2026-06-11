/*
 * SPDX-FileCopyrightText: Tapish Narwal
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/tuple/STLTuple.hpp>

#include <tuple>
#include <utility>

namespace pmacc
{
    namespace memory
    {
        namespace tuple
        {
            namespace detail
            {
                template<typename TFunc, typename TPmaccTuple, std::size_t... Is>
                HDINLINE constexpr decltype(auto) applyImpl(TFunc&& f, TPmaccTuple&& t, std::index_sequence<Is...>)
                {
                    return std::forward<TFunc>(f)(get<Is>(std::forward<TPmaccTuple>(t))...);
                }

                template<typename TFunc, typename TPmaccTuple, std::size_t... Is>
                HDINLINE constexpr decltype(auto) applyEnumerateImpl(
                    TFunc&& f,
                    TPmaccTuple&& t,
                    std::index_sequence<Is...>)
                {
                    return std::forward<TFunc>(f)(tuple::make_tuple(
                        std::integral_constant<std::size_t, Is>{},
                        get<Is>(std::forward<TPmaccTuple>(t)))...);
                }

            } // namespace detail

            // takes pmacc::memory::tuple::Tuple
            template<typename TFunc, typename TPmaccTuple>
            HDINLINE constexpr decltype(auto) apply(TFunc&& f, TPmaccTuple&& t)
            {
                return detail::applyImpl(
                    std::forward<TFunc>(f),
                    std::forward<TPmaccTuple>(t),
                    std::make_index_sequence<tuple_size_v<TPmaccTuple>>{});
            }

            // takes pmacc::memory::tuple::Tuple
            template<typename TFunc, typename TPmaccTuple>
            HDINLINE constexpr decltype(auto) applyEnumerate(TFunc&& f, TPmaccTuple&& t)
            {
                return detail::applyEnumerateImpl(
                    std::forward<TFunc>(f),
                    std::forward<TPmaccTuple>(t),
                    std::make_index_sequence<tuple_size_v<TPmaccTuple>>{});
            }

            namespace detail
            {
                template<size_t... Is, typename... Args, typename Functor>
                constexpr auto tupleMapHelper(
                    std::index_sequence<Is...>,
                    std::tuple<Args...> const& tuple,
                    Functor&& functor)
                {
                    return tuple::make_tuple(std::forward<Functor>(functor)(std::get<Is>(tuple))...);
                }
            } // namespace detail

            /**
             * @brief create a new tuple from the return value of a functor applied on all arguments of a tuple
             */
            template<typename... Args, typename Functor>
            constexpr auto tupleMap(std::tuple<Args...> const& tuple, Functor&& functor)
            {
                return detail::tupleMapHelper(
                    std::make_index_sequence<sizeof...(Args)>{},
                    tuple,
                    std::forward<Functor>(functor));
            }

            /**
             * @brief Converts a std::tuple into a pmacc @ref Tuple
             * @note Host-only. std::tuples aren't trivially copyable anyway
             */
            template<typename... Ts>
            constexpr auto fromStlTuple(std::tuple<Ts...> const& t)
            {
                return std::apply([](auto&&... args) { return make_tuple(std::forward<decltype(args)>(args)...); }, t);
            }

            template<typename... Ts>
            constexpr auto fromStlTuple(std::tuple<Ts...>&& t)
            {
                return std::apply(
                    [](auto&&... args) { return make_tuple(std::forward<decltype(args)>(args)...); },
                    std::move(t));
            }

        } // namespace tuple
    } // namespace memory
} // namespace pmacc
