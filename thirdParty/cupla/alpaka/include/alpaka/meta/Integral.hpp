/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

namespace alpaka::meta
{
    //! The trait is true if all values of TSubset are contained in TSuperset.
    template<typename TSuperset, typename TSubset>
    using IsIntegralSuperset = std::integral_constant<
        bool,
        std::is_integral_v<TSuperset> && std::is_integral_v<TSubset>
            && (
                // If the signdness is equal, the sizes have to be greater or equal to be a superset.
                ((std::is_unsigned_v<TSuperset>
                  == std::is_unsigned_v<TSubset>) &&(sizeof(TSuperset) >= sizeof(TSubset)))
                // If the signdness is non-equal, the superset has to have at least one bit more.
                || ((std::is_unsigned_v<TSuperset> != std::is_unsigned_v<TSubset>) &&(
                    sizeof(TSuperset) > sizeof(TSubset))))>;

    //! The type that has the higher max value.
    template<typename T0, typename T1>
    using HigherMax = std::conditional_t<
        (sizeof(T0) > sizeof(T1)),
        T0,
        std::conditional_t<((sizeof(T0) == sizeof(T1)) && std::is_unsigned_v<T0> && std::is_signed_v<T1>), T0, T1>>;

    //! The type that has the lower max value.
    template<typename T0, typename T1>
    using LowerMax = std::conditional_t<
        (sizeof(T0) < sizeof(T1)),
        T0,
        std::conditional_t<((sizeof(T0) == sizeof(T1)) && std::is_signed_v<T0> && std::is_unsigned_v<T1>), T0, T1>>;

    //! The type that has the higher min value. If both types have the same min value, the type with the wider
    //! range is chosen.
    template<typename T0, typename T1>
    using HigherMin = std::conditional_t<
        (std::is_unsigned_v<T0> == std::is_unsigned_v<T1>),
        std::conditional_t<
            std::is_unsigned_v<T0>,
            std::conditional_t<(sizeof(T0) < sizeof(T1)), T1, T0>,
            std::conditional_t<(sizeof(T0) < sizeof(T1)), T0, T1>>,
        std::conditional_t<std::is_unsigned_v<T0>, T0, T1>>;

    //! The type that has the lower min value. If both types have the same min value, the type with the wider range
    //! is chosen.
    template<typename T0, typename T1>
    using LowerMin = std::conditional_t<
        (std::is_unsigned_v<T0> == std::is_unsigned_v<T1>),
        std::conditional_t<(sizeof(T0) > sizeof(T1)), T0, T1>,
        std::conditional_t<std::is_signed_v<T0>, T0, T1>>;
} // namespace alpaka::meta
