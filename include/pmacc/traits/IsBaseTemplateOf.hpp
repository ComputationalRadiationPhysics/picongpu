/* Copyright 2020-2021 Sergei Bastrakov
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
            static constexpr std::true_type test(const T_Base<T_Args...>*);
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
