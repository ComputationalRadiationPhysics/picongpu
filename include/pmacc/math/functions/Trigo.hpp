/* Copyright 2024 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
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
