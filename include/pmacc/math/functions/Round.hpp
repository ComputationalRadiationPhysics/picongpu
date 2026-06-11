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
    //! Computes the smallest integer value not less than arg.
    ALPAKA_UNARY_MATH_FN(ceil, alpaka::math::ConceptMathCeil, Ceil)

    //! Computes the largest integer value not greater than arg.
    ALPAKA_UNARY_MATH_FN(floor, alpaka::math::ConceptMathFloor, Floor)

    //! Computes the nearest integer not greater in magnitude than arg.
    ALPAKA_UNARY_MATH_FN(trunc, alpaka::math::ConceptMathTrunc, Trunc)

    /** Computes the nearest integer value to arg (in floating-point format).
     *
     * Rounding halfway cases away from zero, regardless of the current rounding mode.
     */
    ALPAKA_UNARY_MATH_FN(round, alpaka::math::ConceptMathRound, Round)

    /** Computes the nearest integer value to arg (in integer format).
     *
     * Rounding halfway cases away from zero, regardless of the current rounding mode.
     */
    ALPAKA_UNARY_MATH_FN(lround, alpaka::math::ConceptMathRound, Lround)

    /** Computes the nearest integer value to arg (in integer format).
     *
     * Rounding halfway cases away from zero, regardless of the current rounding mode.
     */
    ALPAKA_UNARY_MATH_FN(llround, alpaka::math::ConceptMathRound, Llround)
} // namespace pmacc::math
