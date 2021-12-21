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

    //! Computes the smallest integer value not less than arg.
    CUPLA_UNARY_MATH_FN( ceil, alpaka::math::ConceptMathCeil, Ceil )

    //! Computes the largest integer value not greater than arg.
    CUPLA_UNARY_MATH_FN( floor, alpaka::math::ConceptMathFloor, Floor )

    //! Computes the nearest integer not greater in magnitude than arg.
    CUPLA_UNARY_MATH_FN( trunc, alpaka::math::ConceptMathTrunc, Trunc )

    /** Computes the nearest integer value to arg (in floating-point format).
     *
     * Rounding halfway cases away from zero, regardless of the current rounding mode.
     */
    CUPLA_UNARY_MATH_FN( round, alpaka::math::ConceptMathRound, Round )

    /** Computes the nearest integer value to arg (in integer format).
     *
     * Rounding halfway cases away from zero, regardless of the current rounding mode.
     */
    CUPLA_UNARY_MATH_FN( lround, alpaka::math::ConceptMathRound, Lround )

    /** Computes the nearest integer value to arg (in integer format).
     *
     * Rounding halfway cases away from zero, regardless of the current rounding mode.
     */
    CUPLA_UNARY_MATH_FN( llround, alpaka::math::ConceptMathRound, Llround )

} // namespace math
} // namespace device
} // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namespace cupla
