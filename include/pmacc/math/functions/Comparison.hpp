/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/math/functions/Common.hpp"

#include <alpaka/alpaka.hpp>

namespace pmacc::math
{
    //! Calculates the smaller value of two arguments.
    ALPAKA_BINARY_MATH_FN(min, alpaka::math::ConceptMathMin, Min)

    //! Calculates the larger value of two arguments.
    ALPAKA_BINARY_MATH_FN(max, alpaka::math::ConceptMathMax, Max)
} // namespace pmacc::math
