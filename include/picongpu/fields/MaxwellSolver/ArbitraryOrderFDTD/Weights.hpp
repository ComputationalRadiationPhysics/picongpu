/*
 * SPDX-FileCopyrightText: Klaus Steiniger, Sergei Bastrakov, Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/algorithms/math.hpp>

#include <cmath>
#include <cstdint>

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
                        namespace powSpace = pmacc::math;
                        // Set initial value
                        weights[0] = 4.0_X * T_neighbors
                                     * powSpace::cPow(
                                         (factorial(2 * T_neighbors)
                                          / float_X(
                                              powSpace::cPow(2.0_X, 2u * T_neighbors)
                                              * powSpace::cPow(factorial(T_neighbors), 2u))),
                                         2u);

                        // Compute all other values
                        for(uint32_t l = 1u; l < T_neighbors; ++l)
                        {
                            weights[l] = -1.0_X * powSpace::cPow(float_X(l) - 0.5_X, 2u) * (T_neighbors - l)
                                         / float_X(T_neighbors + l) / float_X(powSpace::cPow(float_X(l) + 0.5_X, 2u))
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
