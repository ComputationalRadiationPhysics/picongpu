/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/meta/Set.hpp>

#include <boost/predef.h>

#include <type_traits>
#include <cstddef>

namespace alpaka
{
    namespace meta
    {
        //#############################################################################
        // This could be replaced with c++14 std::IntegerSequence if we raise the minimum.
        template<
            typename T,
            T... Tvals>
        struct IntegerSequence
        {
            static_assert(std::is_integral<T>::value, "IntegerSequence<T, I...> requires T to be an integral type.");

            using type = IntegerSequence<T, Tvals...>;
            using value_type = T;

            ALPAKA_FN_HOST_ACC static auto size() noexcept
            -> std::size_t
            {
                return (sizeof...(Tvals));
            }
        };

        namespace detail
        {
            //#############################################################################
            template<
                typename TDstType,
                typename TIntegerSequence>
            struct ConvertIntegerSequence;
            //#############################################################################
            template<
                typename TDstType,
                typename T,
                T... Tvals>
            struct ConvertIntegerSequence<
                TDstType,
                IntegerSequence<T, Tvals...>>
            {
                using type = IntegerSequence<TDstType, static_cast<TDstType>(Tvals)...>;
            };
        }
        //#############################################################################
        template<
            typename TDstType,
            typename TIntegerSequence>
        using ConvertIntegerSequence = typename detail::ConvertIntegerSequence<TDstType, TIntegerSequence>::type;

        namespace detail
        {
            //#############################################################################
            template<
                template<typename...> class TList,
                typename T,
                template<T> class TOp,
                typename TIntegerSequence>
            struct TransformIntegerSequence;
            //#############################################################################
            template<
                template<typename...> class TList,
                typename T,
                template<T> class TOp,
                T... Tvals>
            struct TransformIntegerSequence<
                TList,
                T,
                TOp,
                IntegerSequence<T, Tvals...>>
            {
                using type =
                    TList<
                        TOp<Tvals>...>;
            };
        }
        //#############################################################################
        template<
            template<typename...> class TList,
            typename T,
            template<T> class TOp,
            typename TIntegerSequence>
        using TransformIntegerSequence = typename detail::TransformIntegerSequence<TList, T, TOp, TIntegerSequence>::type;

        namespace detail
        {
            //#############################################################################
            template<bool TisSizeNegative, bool TbIsBegin, typename T, T Tbegin, typename TIntCon, typename TIntSeq>
            struct MakeIntegerSequenceHelper
            {
                static_assert(!TisSizeNegative, "MakeIntegerSequence<T, N> requires N to be non-negative.");
            };
            //#############################################################################
            template<typename T, T Tbegin, T... Tvals>
            struct MakeIntegerSequenceHelper<false, true, T, Tbegin, std::integral_constant<T, Tbegin>, IntegerSequence<T, Tvals...> > :
                IntegerSequence<T, Tvals...>
            {};
            //#############################################################################
            template<typename T, T Tbegin, T TIdx, T... Tvals>
            struct MakeIntegerSequenceHelper<false, false, T, Tbegin, std::integral_constant<T, TIdx>, IntegerSequence<T, Tvals...> > :
                MakeIntegerSequenceHelper<false, TIdx == (Tbegin+1), T, Tbegin, std::integral_constant<T, TIdx - 1>, IntegerSequence<T, TIdx - 1, Tvals...> >
            {};
        }

        //#############################################################################
        template<typename T, T Tbegin, T Tsize>
        using MakeIntegerSequenceOffset = typename detail::MakeIntegerSequenceHelper<(Tsize < 0), (Tsize == 0), T, Tbegin, std::integral_constant<T, Tbegin+Tsize>, IntegerSequence<T> >::type;

        //#############################################################################
        template<typename T, T Tsize>
        using MakeIntegerSequence = MakeIntegerSequenceOffset<T, 0u, Tsize>;


        //#############################################################################
        template<
            std::size_t... Tvals>
        using IndexSequence = IntegerSequence<std::size_t, Tvals...>;

        //#############################################################################
        template<
            typename T,
            T Tbegin,
            T Tsize>
        using MakeIndexSequenceOffset = MakeIntegerSequenceOffset<std::size_t, Tbegin, Tsize>;

        //#############################################################################
        template<
            std::size_t Tsize>
        using MakeIndexSequence = MakeIntegerSequence<std::size_t, Tsize>;

        //#############################################################################
        template<
            typename... Ts>
        using IndexSequenceFor = MakeIndexSequence<sizeof...(Ts)>;


        //#############################################################################
        //! Checks if the integral values are unique.
        template<
            typename T,
            T... Tvals>
        struct IntegralValuesUnique
        {
            static constexpr bool value = meta::IsParameterPackSet<std::integral_constant<T, Tvals>...>::value;
        };

        //#############################################################################
        //! Checks if the values in the index sequence are unique.
        template<
            typename TIntegerSequence>
        struct IntegerSequenceValuesUnique;
        //#############################################################################
        //! Checks if the values in the index sequence are unique.
        template<
            typename T,
            T... Tvals>
        struct IntegerSequenceValuesUnique<
            IntegerSequence<T, Tvals...>>
        {
            static constexpr bool value = IntegralValuesUnique<T, Tvals...>::value;
        };

        //#############################################################################
        //! Checks if the integral values are within the given range.
        template<
            typename T,
            T Tmin,
            T Tmax,
            T... Tvals>
        struct IntegralValuesInRange;
        //#############################################################################
        //! Checks if the integral values are within the given range.
        template<
            typename T,
            T Tmin,
            T Tmax>
        struct IntegralValuesInRange<
            T,
            Tmin,
            Tmax>
        {
            static constexpr bool value = true;
        };
        //#############################################################################
        //! Checks if the integral values are within the given range.
        template<
            typename T,
            T Tmin,
            T Tmax,
            T I,
            T... Tvals>
        struct IntegralValuesInRange<
            T,
            Tmin,
            Tmax,
            I,
            Tvals...>
        {
            static constexpr bool value = (I >= Tmin) && (I <=Tmax) && IntegralValuesInRange<T, Tmin, Tmax, Tvals...>::value;
        };

        //#############################################################################
        //! Checks if the values in the index sequence are within the given range.
        template<
            typename TIntegerSequence,
            typename T,
            T Tmin,
            T Tmax>
        struct IntegerSequenceValuesInRange;
        //#############################################################################
        //! Checks if the values in the index sequence are within the given range.
        template<
            typename T,
            T... Tvals,
            T Tmin,
            T Tmax>
        struct IntegerSequenceValuesInRange<
            IntegerSequence<T, Tvals...>,
            T,
            Tmin,
            Tmax>
        {
            static constexpr bool value = IntegralValuesInRange<T, Tmin, Tmax, Tvals...>::value;
        };
    }
}
