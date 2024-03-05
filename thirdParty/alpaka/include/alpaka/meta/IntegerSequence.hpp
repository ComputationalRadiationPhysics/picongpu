/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/meta/Set.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace alpaka::meta
{
    namespace detail
    {
        template<typename TDstType, typename TIntegerSequence>
        struct ConvertIntegerSequence;

        template<typename TDstType, typename T, T... Tvals>
        struct ConvertIntegerSequence<TDstType, std::integer_sequence<T, Tvals...>>
        {
            using type = std::integer_sequence<TDstType, static_cast<TDstType>(Tvals)...>;
        };
    } // namespace detail

    template<typename TDstType, typename TIntegerSequence>
    using ConvertIntegerSequence = typename detail::ConvertIntegerSequence<TDstType, TIntegerSequence>::type;

    namespace detail
    {
        template<bool TisSizeNegative, bool TbIsBegin, typename T, T Tbegin, typename TIntCon, typename TIntSeq>
        struct MakeIntegerSequenceHelper
        {
            static_assert(!TisSizeNegative, "MakeIntegerSequence<T, N> requires N to be non-negative.");
        };

        template<typename T, T Tbegin, T... Tvals>
        struct MakeIntegerSequenceHelper<
            false,
            true,
            T,
            Tbegin,
            std::integral_constant<T, Tbegin>,
            std::integer_sequence<T, Tvals...>>
        {
            using type = std::integer_sequence<T, Tvals...>;
        };

        template<typename T, T Tbegin, T TIdx, T... Tvals>
        struct MakeIntegerSequenceHelper<
            false,
            false,
            T,
            Tbegin,
            std::integral_constant<T, TIdx>,
            std::integer_sequence<T, Tvals...>>
        {
            using type = typename MakeIntegerSequenceHelper<
                false,
                TIdx == (Tbegin + 1),
                T,
                Tbegin,
                std::integral_constant<T, TIdx - 1>,
                std::integer_sequence<T, TIdx - 1, Tvals...>>::type;
        };
    } // namespace detail

    template<typename T, T Tbegin, T Tsize>
    using MakeIntegerSequenceOffset = typename detail::MakeIntegerSequenceHelper<
        (Tsize < 0),
        (Tsize == 0),
        T,
        Tbegin,
        std::integral_constant<T, Tbegin + Tsize>,
        std::integer_sequence<T>>::type;

    //! Checks if the integral values are unique.
    template<typename T, T... Tvals>
    struct IntegralValuesUnique
    {
        static constexpr bool value = meta::IsParameterPackSet<std::integral_constant<T, Tvals>...>::value;
    };

    //! Checks if the values in the index sequence are unique.
    template<typename TIntegerSequence>
    struct IntegerSequenceValuesUnique;

    //! Checks if the values in the index sequence are unique.
    template<typename T, T... Tvals>
    struct IntegerSequenceValuesUnique<std::integer_sequence<T, Tvals...>>
    {
        static constexpr bool value = IntegralValuesUnique<T, Tvals...>::value;
    };

    //! Checks if the integral values are within the given range.
    template<typename T, T Tmin, T Tmax, T... Tvals>
    struct IntegralValuesInRange;

    //! Checks if the integral values are within the given range.
    template<typename T, T Tmin, T Tmax>
    struct IntegralValuesInRange<T, Tmin, Tmax>
    {
        static constexpr bool value = true;
    };

    //! Checks if the integral values are within the given range.
    template<typename T, T Tmin, T Tmax, T I, T... Tvals>
    struct IntegralValuesInRange<T, Tmin, Tmax, I, Tvals...>
    {
        static constexpr bool value
            = (I >= Tmin) && (I <= Tmax) && IntegralValuesInRange<T, Tmin, Tmax, Tvals...>::value;
    };

    //! Checks if the values in the index sequence are within the given range.
    template<typename TIntegerSequence, typename T, T Tmin, T Tmax>
    struct IntegerSequenceValuesInRange;

    //! Checks if the values in the index sequence are within the given range.
    template<typename T, T... Tvals, T Tmin, T Tmax>
    struct IntegerSequenceValuesInRange<std::integer_sequence<T, Tvals...>, T, Tmin, Tmax>
    {
        static constexpr bool value = IntegralValuesInRange<T, Tmin, Tmax, Tvals...>::value;
    };
} // namespace alpaka::meta
