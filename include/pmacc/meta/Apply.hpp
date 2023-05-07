/* Copyright 2022 Bernhard Manfred Gruber
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "Mp11.hpp"

namespace pmacc
{
    namespace detail
    {
        template<typename E, typename... Args>
        struct ReplacePlaceholdersImpl
        {
            using type = E;
        };
        template<std::size_t I, typename... Args>
        struct ReplacePlaceholdersImpl<mp_arg<I>, Args...>
        {
            using type = mp_at_c<mp_list<Args...>, I>;
        };

        template<template<typename...> typename E, typename... Ts, typename... Args>
        struct ReplacePlaceholdersImpl<E<Ts...>, Args...>
        {
            using type = E<typename ReplacePlaceholdersImpl<Ts, Args...>::type...>;
            // nested ::type of E to mimic mpl::apply
            // using type = typename E<typename ReplacePlaceholdersImpl<Ts, Args...>::type...>::type;
        };
    } // namespace detail

    template<typename Expression, typename... Args>
    using ReplacePlaceholders = typename detail::ReplacePlaceholdersImpl<Expression, Args...>::type;

    namespace detail
    {
        template<typename T>
        inline constexpr bool isPlaceholderExpression = false;

        template<std::size_t I>
        inline constexpr bool isPlaceholderExpression<mp_arg<I>> = true;

        template<template<typename...> typename L, typename... Ts>
        inline constexpr bool isPlaceholderExpression<L<Ts...>> = (isPlaceholderExpression<Ts> || ... || false);

        template<typename Expression, bool IsPlaceholderExpression = isPlaceholderExpression<Expression>>
        struct ApplyImpl : Expression
        {
        };

        template<typename Expression>
        struct ApplyImpl<Expression, true>
        {
            template<typename... Args>
            using fn = ReplacePlaceholders<Expression, Args...>;
        };
    } // namespace detail

    /// Equivalent of boost::mpl::apply. If Expression contains placeholders (such as _1 etc.), the placeholders are
    /// replaced by Args via pmacc::ReplacePlaceholders. If there are no placeholders, Expression is treated as a
    /// quoted meta function and the Expression::fn<Args...> is returned.
    template<typename Expression, typename... Args>
    using Apply = typename detail::ApplyImpl<Expression>::template fn<Args...>;
} // namespace pmacc
