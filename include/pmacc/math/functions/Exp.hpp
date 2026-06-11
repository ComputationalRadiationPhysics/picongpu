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
    //! Computes e (Euler's number, 2.7182818...) raised to the given power.
    ALPAKA_UNARY_MATH_FN(exp, alpaka::math::ConceptMathExp, Exp)
} // namespace pmacc::math
