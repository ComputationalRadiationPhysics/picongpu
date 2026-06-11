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
    //! Computes the natural (base e) logarithm.
    ALPAKA_UNARY_MATH_FN(log, alpaka::math::ConceptMathLog, Log)

    //! Computes the logarithm to the base of 2.
    ALPAKA_UNARY_MATH_FN(log2, alpaka::math::ConceptMathLog2, Log2)

    //! Computes the logarithm to the base of 10.
    ALPAKA_UNARY_MATH_FN(log10, alpaka::math::ConceptMathLog10, Log10)
} // namespace pmacc::math
