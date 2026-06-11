/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Richard Pausch, Axel Huebl, Alexander Debus
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    namespace math
    {
        template<typename ArgType, typename SinType, typename CosType>
        struct SinCos;

        template<typename Type>
        struct Sinc;

        template<typename ArgType, typename SinType, typename CosType>
        HDINLINE typename SinCos<ArgType, SinType, CosType>::result sincos(
            ArgType arg,
            SinType& sinValue,
            CosType& cosValue)
        {
            return SinCos<ArgType, SinType, CosType>()(arg, sinValue, cosValue);
        }

        template<typename T1>
        HDINLINE typename Sinc<T1>::result sinc(const T1& value)
        {
            return Sinc<T1>()(value);
        }

    } /* namespace math */
} /* namespace pmacc */

#include "pmacc/algorithms/math/doubleMath/trigo.tpp"
#include "pmacc/algorithms/math/floatMath/trigo.tpp"
