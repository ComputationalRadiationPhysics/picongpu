/* Copyright 2015-2021 Benjamin Worpitz, Richard Pausch
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

#include "pmacc/types.hpp"
#include <cmath>


namespace pmacc
{
    namespace math
    {
        template<>
        struct Min<double, double>
        {
            typedef double result;

            HDINLINE double operator()(double value1, double value2)
            {
                return ::fmin(value1, value2);
            }
        };

        template<>
        struct Max<double, double>
        {
            typedef double result;

            HDINLINE double operator()(double value1, double value2)
            {
                return ::fmax(value1, value2);
            }
        };

    } // namespace math
} // namespace pmacc
