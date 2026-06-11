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
        template<typename Type1, typename Type2>
        struct Dot;

        template<typename T1, typename T2>
        HDINLINE typename Dot<T1, T2>::result dot(const T1& value, const T2& value2)
        {
            return Dot<T1, T2>()(value, value2);
        }
    } // namespace math
} // namespace pmacc
