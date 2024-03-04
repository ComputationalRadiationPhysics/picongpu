/* Copyright 2023 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

namespace alpaka::meta
{
    namespace detail
    {
        template<typename T>
        struct NonZeroImpl : std::false_type
        {
        };

        template<typename T, T TValue>
        struct NonZeroImpl<std::integral_constant<T, TValue>> : std::bool_constant<TValue != static_cast<T>(0)>
        {
        };
    } // namespace detail

    template<typename T>
    using NonZero = typename detail::NonZeroImpl<T>;

} // namespace alpaka::meta
