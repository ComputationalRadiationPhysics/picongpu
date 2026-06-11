/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera
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
        struct Norm;

        template<typename T1>
        HDINLINE typename Norm<T1>::result norm(const T1& value)
        {
            return Norm<T1>()(value);
        }
    } // namespace math
} // namespace pmacc

#include "pmacc/algorithms/math/doubleMath/norm.tpp"
#include "pmacc/algorithms/math/floatMath/norm.tpp"
