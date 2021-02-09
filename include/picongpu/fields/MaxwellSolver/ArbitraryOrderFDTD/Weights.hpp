/* Copyright 2020-2021 Klaus Steiniger, Sergei Bastrakov
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

#pragma once

#include <cstdint>
#include <cmath>
#include "picongpu/plugins/radiation/utilities.hpp"


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            namespace aoFDTD
            {
                /** Compute weights of finite differences in
                 *
                 * @tparam T_neighbors Number of neighbors used to calculate
                 *                     the derivative from finite differences.
                 *                     Order of derivative approximation is
                 *                     2 * T_neighbors
                 */
                template<uint32_t T_neighbors>
                struct AOFDTDWeights
                {
                    HDINLINE constexpr AOFDTDWeights()
                    {
                        namespace powSpace = ::picongpu::plugins::radiation::util;
                        // Set initial value
                        weights[0] = 4.0_X * T_neighbors
                            * powSpace::pow(
                                         (factorial(2 * T_neighbors)
                                          / float_X(
                                              powSpace::pow(2.0_X, 2 * T_neighbors)
                                              * powSpace::pow(factorial(T_neighbors), 2))),
                                         2);

                        // Compute all other values
                        for(uint32_t l = 1u; l < T_neighbors; ++l)
                        {
                            weights[l] = -1.0_X * powSpace::pow(float_X(l) - 0.5_X, 2) * (T_neighbors - l)
                                / float_X(T_neighbors + l) / float_X(powSpace::pow(float_X(l) + 0.5_X, 2))
                                * weights[l - 1];
                        }
                    }

                    HDINLINE constexpr float_X operator[](uint32_t const l) const
                    {
                        PMACC_ASSERT_MSG(l < T_neighbors, "NUMBER_OF_COEFFICIENTS_IS_LIMITED_BY_NUMBER_OF_NEIGHBORS");
                        return weights[l];
                    }

                private:
                    HDINLINE constexpr uint32_t factorial(uint32_t const n) const
                    {
                        return n <= 1u ? 1u : (n * factorial(n - 1u));
                    }

                    float_X weights[T_neighbors];
                };
            } // namespace aoFDTD
        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu
