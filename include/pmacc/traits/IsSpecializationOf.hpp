/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/* Copyright 2025 Tapish Narwal
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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
