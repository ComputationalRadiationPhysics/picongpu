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
    //! Computes the error function.
    ALPAKA_UNARY_MATH_FN(erf, alpaka::math::ConceptMathErf, Erf)
} // namespace pmacc::math
