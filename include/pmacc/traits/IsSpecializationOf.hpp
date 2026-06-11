/*
 * SPDX-FileCopyrightText: Tapish Narwal
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <type_traits>

namespace pmacc
{
    namespace traits
    {

        /**
         * Type trait to check if a type is a specialization of a template
         * Similar to P2078 - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2098r0.pdf
         * Note that this cant be used with template types which have NTTPs
         * To fix this limitation we need PR1985 Universal Template Parameters
         * https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p1985r3.pdf
         */

        template<typename, template<typename...> typename>
        struct IsSpecializationOf : std::false_type
        {
        };

        template<template<typename...> typename Template, typename... Args>
        struct IsSpecializationOf<Template<Args...>, Template> : std::true_type
        {
        };

    } // namespace traits

    template<typename T, template<typename...> typename Template>
    inline constexpr bool isSpecializationOf_v = traits::IsSpecializationOf<T, Template>::value;

    namespace concepts
    {
        template<typename T, template<typename...> typename Template>
        concept SpecializationOf = isSpecializationOf_v<T, Template>;

    } // namespace concepts

} // namespace pmacc
