/* Copyright 2019-2020 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file */

#pragma once

#include <pmacc/algorithms/math.hpp>


struct numericalDifferentation
{
    template<typename T_Argument>
    static float_X scaledChebyshevNodes(uint8_t k, uint8_t N, T_Argument centralE, T_Argument deltaE)
    {
        /** returns the k-th of N scaled chebyshev nodes
         *
         * @Param k ... which chebyshev Nodes is to be used
         * @Param N ... how many chebyshev nodes are required
         *
         * BEWARE: k = 1, ..., N, k=0 is not allowed
         * BEWARE: the highest argument node corresponds to the lowest k value
         *
         * see https://en.wikipedia.org/wiki/Chebyshev_nodes for more information
         */

        // check for bounds on k
        PMACC_ASSERT_MSG(k >= 1 and k <= N, "chebyshev nodes are defined only for 1 <= k <= N");

        math::cos<float_X>((2 * k - 1) / (2_X * N) * math::Pi::value)
                * static_cast<float_X>(deltaE) / 2.0
            + static_cast<float_X>(centralE);
    }