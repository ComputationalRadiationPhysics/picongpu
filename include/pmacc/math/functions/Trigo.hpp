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
    //! Computes the sine (measured in radians).
    ALPAKA_UNARY_MATH_FN(sin, alpaka::math::ConceptMathSin, Sin)

    //! Computes the cosine (measured in radians).
    ALPAKA_UNARY_MATH_FN(cos, alpaka::math::ConceptMathCos, Cos)

    //! Computes the tangent (measured in radians).
    ALPAKA_UNARY_MATH_FN(tan, alpaka::math::ConceptMathTan, Tan)

    //! Computes the principal value of the arc sine.
    ALPAKA_UNARY_MATH_FN(asin, alpaka::math::ConceptMathAsin, Asin)

    //! Computes the principal value of the arc cosine.
    ALPAKA_UNARY_MATH_FN(acos, alpaka::math::ConceptMathAcos, Acos)

    //! Computes the principal value of the arc tangent.
    ALPAKA_UNARY_MATH_FN(atan, alpaka::math::ConceptMathAtan, Atan)

    //! Computes the arc tangent of y/x using the signs of arguments to determine the correct quadrant.
    ALPAKA_BINARY_MATH_FN(atan2, alpaka::math::ConceptMathAtan2, Atan2)

    //! Computes the hyperbolic sine.
    ALPAKA_UNARY_MATH_FN(sinh, alpaka::math::ConceptMathSinh, Sinh)

    //! Computes the hyperbolic cosine.
    ALPAKA_UNARY_MATH_FN(cosh, alpaka::math::ConceptMathCosh, Cosh)

    //! Computes the hyperbolic tangent.
    ALPAKA_UNARY_MATH_FN(tanh, alpaka::math::ConceptMathTanh, Tanh)

    //! Computes the hyperbolic arc sine.
    ALPAKA_UNARY_MATH_FN(asinh, alpaka::math::ConceptMathAsin, Asinh)

    //! Computes the hyperbolic arc cosine.
    ALPAKA_UNARY_MATH_FN(acosh, alpaka::math::ConceptMathAcosh, Acosh)

    //! Computes the hyperbolic arc tangent.
    ALPAKA_UNARY_MATH_FN(atanh, alpaka::math::ConceptMathAtanh, Atanh)
} // namespace pmacc::math
