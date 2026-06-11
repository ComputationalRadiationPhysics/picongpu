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
        struct Cross;

        template<typename T1, typename T2>
        HDINLINE typename Cross<T1, T2>::result cross(const T1& value, const T2& value2)
        {
            return Cross<T1, T2>()(value, value2);
        }
    } // namespace math
} // namespace pmacc
