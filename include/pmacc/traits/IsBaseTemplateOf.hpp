/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <type_traits>

namespace pmacc
{
    namespace traits
    {
        /** Check if a type inherits the given class template (with any arguments)
         *
         * This is basically a version of std::is_base_of but for class template as base.
         * Based on Stack Overflow post:
         *   source: https://stackoverflow.com/a/34672753
         *   author: rmawatson
         *   date: Aug 23 '18
         *
         * @tparam T_Base base template (itself, without arguments)
         * @tparam T_Derived derived type to check
         * @treturn ::type std::true_type or std::false_type
         */
        template<template<typename...> class T_Base, typename T_Derived>
        struct IsBaseTemplateOf
        {
            template<typename... T_Args>
            static constexpr std::true_type test(T_Base<T_Args...> const*);
            static constexpr std::false_type test(...);
            using type = decltype(test(std::declval<T_Derived*>()));
        };

        /** Helper alias for IsBaseTemplateOf<...>::type
         *
         * @tparam T_Base base template (itself, without arguments)
         * @tparam T_Derived derived type to check
         * @treturn std::true_type or std::false_type
         */
        template<template<typename...> class T_Base, typename T_Derived>
        using IsBaseTemplateOf_t = typename IsBaseTemplateOf<T_Base, T_Derived>::type;

    } // namespace traits
} // namespace pmacc
