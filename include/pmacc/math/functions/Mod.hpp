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
    //! Computes the floating-point remainder of the division operation x/y.
    ALPAKA_BINARY_MATH_FN(fmod, alpaka::math::ConceptMathFmod, Fmod)

    //! Computes the IEEE remainder of the floating point division operation x/y.
    ALPAKA_BINARY_MATH_FN(remainder, alpaka::math::ConceptMathRemainder, Remainder)
} // namespace pmacc::math
