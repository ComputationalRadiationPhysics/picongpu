/*
 * SPDX-FileCopyrightText: Heiko Burau
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    namespace math
    {
        template<typename Type>
        struct Modf;

        template<typename T>
        HDINLINE typename Modf<T>::result modf(T value, T* intpart)
        {
            return Modf<T>()(value, intpart);
        }

    } // namespace math
} // namespace pmacc

#include "pmacc/algorithms/math/doubleMath/modf.tpp"
#include "pmacc/algorithms/math/floatMath/modf.tpp"
