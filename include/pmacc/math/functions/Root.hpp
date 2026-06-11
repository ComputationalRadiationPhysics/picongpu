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
    //! Computes the square root.
    ALPAKA_UNARY_MATH_FN(sqrt, alpaka::math::ConceptMathSqrt, Sqrt)

    //! Computes the inverse square root.
    ALPAKA_UNARY_MATH_FN(rsqrt, alpaka::math::ConceptMathRsqrt, Rsqrt)

    //! Computes the cubic root.
    ALPAKA_UNARY_MATH_FN(cbrt, alpaka::math::ConceptMathCbrt, Cbrt)
} // namespace pmacc::math
