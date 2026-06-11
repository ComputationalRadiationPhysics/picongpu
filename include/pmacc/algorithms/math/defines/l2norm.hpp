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
        /// definition must be provided by Type
        template<typename Type>
        struct L2norm;

        /** l2norm
         *
         * only defined for vectors
         *
         * @return sqrt(abs(x)^2 + ...)
         */
        template<typename T1>
        HDINLINE typename L2norm<T1>::result l2norm(const T1& value)
        {
            return L2norm<T1>()(value);
        }

        template<typename Type>
        struct L2norm2;

        /** l2norm2
         *
         * only defined for vectors
         *
         * @return abs(x)^2 + ...
         */
        template<typename T1>
        HDINLINE typename L2norm2<T1>::result l2norm2(const T1& value)
        {
            return L2norm2<T1>()(value);
        }
    } // namespace math
} // namespace pmacc
