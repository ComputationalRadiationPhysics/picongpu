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
    //! Computes the absolute value.
    ALPAKA_UNARY_MATH_FN(abs, alpaka::math::ConceptMathAbs, Abs)
} // namespace pmacc::math
