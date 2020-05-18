/* Copyright 2020 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */


#pragma once

#include "cupla/device/math/Common.hpp"
#include "cupla/types.hpp"

namespace cupla
{
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{
inline namespace device
{
inline namespace math
{

    //! Computes the sine (measured in radians).
    CUPLA_UNARY_MATH_FN( sin, alpaka::math::ConceptMathSin, Sin )

    //! Computes the cosine (measured in radians).
    CUPLA_UNARY_MATH_FN( cos, alpaka::math::ConceptMathCos, Cos )

    //! Computes the tangent (measured in radians).
    CUPLA_UNARY_MATH_FN( tan, alpaka::math::ConceptMathTan, Tan )

    //! Computes the principal value of the arc sine.
    CUPLA_UNARY_MATH_FN( asin, alpaka::math::ConceptMathAsin, Asin )

    //! Computes the principal value of the arc cosine.
    CUPLA_UNARY_MATH_FN( acos, alpaka::math::ConceptMathAcos, Acos )

    //! Computes the principal value of the arc tangent.
    CUPLA_UNARY_MATH_FN( atan, alpaka::math::ConceptMathAtan, Atan )

    //! Computes the arc tangent of y/x using the signs of arguments to determine the correct quadrant.
    CUPLA_BINARY_MATH_FN( atan2, alpaka::math::ConceptMathAtan2, Atan2 )

} // namespace math
} // namespace device
} // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namespace cupla
