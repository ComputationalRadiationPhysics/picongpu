/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <type_traits>

namespace alpaka
{
    namespace meta
    {
        //#############################################################################
        //! The trait is true if all values of TSubset are contained in TSuperset.
        template<typename TSuperset, typename TSubset>
        using IsIntegralSuperset = std::integral_constant<
            bool,
            std::is_integral<TSuperset>::value && std::is_integral<TSubset>::value
                && (
                    // If the signdness is equal, the sizes have to be greater or equal to be a superset.
                    ((std::is_unsigned<TSuperset>::value == std::is_unsigned<TSubset>::value)
                     && (sizeof(TSuperset) >= sizeof(TSubset)))
                    // If the signdness is non-equal, the superset has to have at least one bit more.
                    || ((std::is_unsigned<TSuperset>::value != std::is_unsigned<TSubset>::value)
                        && (sizeof(TSuperset) > sizeof(TSubset))))>;

        //#############################################################################
        //! The type that has the higher max value.
        template<typename T0, typename T1>
        using HigherMax = std::conditional_t<
            (sizeof(T0) > sizeof(T1)),
            T0,
            std::conditional_t<
                ((sizeof(T0) == sizeof(T1)) && std::is_unsigned<T0>::value && std::is_signed<T1>::value),
                T0,
                T1>>;

        //#############################################################################
        //! The type that has the lower max value.
        template<typename T0, typename T1>
        using LowerMax = std::conditional_t<
            (sizeof(T0) < sizeof(T1)),
            T0,
            std::conditional_t<
                ((sizeof(T0) == sizeof(T1)) && std::is_signed<T0>::value && std::is_unsigned<T1>::value),
                T0,
                T1>>;

        //#############################################################################
        //! The type that has the higher min value. If both types have the same min value, the type with the wider
        //! range is chosen.
        template<typename T0, typename T1>
        using HigherMin = std::conditional_t<
            (std::is_unsigned<T0>::value == std::is_unsigned<T1>::value),
            std::conditional_t<
                std::is_unsigned<T0>::value,
                std::conditional_t<(sizeof(T0) < sizeof(T1)), T1, T0>,
                std::conditional_t<(sizeof(T0) < sizeof(T1)), T0, T1>>,
            std::conditional_t<std::is_unsigned<T0>::value, T0, T1>>;

        //#############################################################################
        //! The type that has the lower min value. If both types have the same min value, the type with the wider range
        //! is chosen.
        template<typename T0, typename T1>
        using LowerMin = std::conditional_t<
            (std::is_unsigned<T0>::value == std::is_unsigned<T1>::value),
            std::conditional_t<(sizeof(T0) > sizeof(T1)), T0, T1>,
            std::conditional_t<std::is_signed<T0>::value, T0, T1>>;
    } // namespace meta
} // namespace alpaka
