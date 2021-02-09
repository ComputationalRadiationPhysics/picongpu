/* Copyright 2015-2021 Heiko Burau
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

#include "pmacc/math/vector/compile-time/Vector.hpp"

namespace pmacc
{
    namespace math
    {
        namespace CT
        {
            /**
             * @class TwistComponents
             * @brief Twists axes of a compile-time vector.
             * @tparam Vec compile-time vector to be twisted
             * @tparam Axes compile-time vector containing new axes
             *
             * Example:
             *
             * using Orientation_Y = pmacc::math::CT::Int<1,2,0>;
             * using TwistedBlockDim = typename pmacc::math::CT::TwistComponents<BlockDim, Orientation_Y>::type;
             */
            template<typename Vec, typename Axes, int dim = Vec::dim>
            struct TwistComponents;

            template<typename Vec, typename Axes>
            struct TwistComponents<Vec, Axes, DIM2>
            {
                using type = math::CT::Vector<
                    typename Vec::template at<Axes::x::value>::type,
                    typename Vec::template at<Axes::y::value>::type>;
            };

            template<typename Vec, typename Axes>
            struct TwistComponents<Vec, Axes, DIM3>
            {
                using type = math::CT::Vector<
                    typename Vec::template at<Axes::x::value>::type,
                    typename Vec::template at<Axes::y::value>::type,
                    typename Vec::template at<Axes::z::value>::type>;
            };

        } // namespace CT
    } // namespace math
} // namespace pmacc
