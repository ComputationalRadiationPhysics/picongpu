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
                //! Computes the natural (base e) logarithm.
                CUPLA_UNARY_MATH_FN(log, alpaka::math::ConceptMathLog, Log)

#if ALPAKA_VERSION >= BOOST_VERSION_NUMBER(1, 0, 0)
                //! Computes the natural (base 2) logarithm.
                CUPLA_UNARY_MATH_FN(log2, alpaka::math::ConceptMathLog2, Log2)

                //! Computes the natural (base 10) logarithm.
                CUPLA_UNARY_MATH_FN(log10, alpaka::math::ConceptMathLog10, Log10)
#endif

            } // namespace math
        } // namespace device
    } // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namespace cupla
