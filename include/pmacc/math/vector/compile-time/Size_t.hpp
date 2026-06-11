/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/math/vector/Vector.hpp"

#include <cstdint>

#include "pmacc/math/vector/compile-time/Vector.hpp"

namespace pmacc
{
    namespace math
    {
        namespace CT
        {
            /** Compile time size_t vector
             *
             *
             * @tparam x value for x allowed range [0;max size_t value -1]
             * @tparam y value for y allowed range [0;max size_t value -1]
             * @tparam z value for z allowed range [0;max size_t value -1]
             *
             * default parameter is used to distinguish between values given by
             * the user and unset values.
             */
            template<size_t... T_values>
            using Size_t = CT::Vector<std::integral_constant<size_t, T_values>...>;
        } // namespace CT
    } // namespace math
} // namespace pmacc
