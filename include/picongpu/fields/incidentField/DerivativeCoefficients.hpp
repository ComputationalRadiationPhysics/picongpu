/* Copyright 2021 Sergei Bastrakov
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/MaxwellSolver/ArbitraryOrderFDTD/Derivative.hpp"
#include "picongpu/fields/MaxwellSolver/Lehe/Derivative.hpp"

#include <cstdint>


namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace detail
            {
                /** Coefficients for the finite-difference derivative operator along the given axis
                 *
                 * The current implementation assumes a certain form of the operator.
                 * Namely, that Yee grid and central operator is used.
                 * Additionally, we assume that only one "row" of values along T_axis is used.
                 *
                 * It represents a finite-difference derivative in this form (for T_axis == 0, similar for others):
                 * dF/dx (i, j, k) = (1.0 / stepX) * sum(values[p] * (F(i + (p + 1/2), j, k) - F(i - (p + 1/2), j, k));
                 * p = 0, 1, ..., size - 1).
                 * Where F is the incident field and i, j, k can be half-integer according to the Yee grid.
                 *
                 * The default implementation works for the classic 2nd order central derivative on the Yee grid.
                 *
                 * @note We do not make the difference between the forward and backward derivative.
                 * This is becase it is already part of index calculations in the host side of the incident field
                 * solver. Same (no difference) should hold for all specializations.
                 *
                 * @note The aforementioned assumptions are not principle, but taken according to the currently used
                 * solvers. (This is not true for the Lehe solver, @see notes at the corresponding specialization.) The
                 * scheme can be generalized for other grids and non-central derivatives, then a full set of
                 * coefficients will have to be stored, without assuming antisymmetry. In case multiple "rows" is used,
                 * the values array will need to become simDim-dimensional. Such modifications would only affect this
                 * struct and the inner loop in UpdateFunctor::operator().
                 *
                 * @tparam T_DerivativeFunctor derivative functor type along the axis
                 * @tparam T_axis derivative axis, 0 = x, 1 = y, 2 = z
                 */
                template<typename T_DerivativeFunctor, uint32_t T_axis>
                struct DerivativeCoefficients
                {
                    //! Number of coefficients = number of neighbors used in each side along T_axis
                    static constexpr uint32_t size = 1u;

                    /** Normalized coefficient values, actual coefficient = values / step
                     *
                     * The coefficients are for positive-side terms (see equation above).
                     * For the negative side we assume antisymmetry.
                     */
                    float_X values[size] = {1.0_X};
                };

                /** Specialization for the arbitrary-order FDTD derivative operator
                 *
                 * Same as in the general implementation, this is for both forward and backward derivatives.
                 *
                 * @tparam T_lowerNeighbors number of neighbors required in negative
                 *                          direction to calculate field derivative
                 * @tparam T_upperNeighbors number of neighbors required in positive
                 *                          direction to calculate field derivative
                 * @tparam T_neighbors number of neighbors used to calculate
                 *                     the derivative from finite differences.
                 *                     Order of derivative approximation is
                 *                     2 * T_neighbors
                 * @tparam T_axis derivative axis, 0 = x, 1 = y, 2 = z
                 */
                template<uint32_t T_lowerNeighbors, uint32_t T_upperNeighbors, uint32_t T_neighbors, uint32_t T_axis>
                struct DerivativeCoefficients<
                    maxwellSolver::aoFDTD::detail::
                        GeneralAofdtdDerivative<T_lowerNeighbors, T_upperNeighbors, T_neighbors, T_axis>,
                    T_axis>
                {
                    //! Number of coefficients = number of neighbors used in each side
                    static constexpr uint32_t size = T_neighbors;

                    //! Normalized coefficient values, actual coefficient = values / step
                    float_X values[size];

                    //! Instantiate and initialize derivative coefficients
                    HDINLINE DerivativeCoefficients()
                    {
                        // E.g. for 4th-order FDTD (T_neighbors = 2) values[0] = 27/24, values[1] = -1/24
                        maxwellSolver::aoFDTD::AOFDTDWeights<size> weights;
                        for(uint32_t i = 0; i < size; i++)
                            values[i] = weights[i];
                    }
                };

                /** Specialization for the Lehe derivative along the Cherenkov-free direction
                 *
                 * The Lehe derivative uses terms that differ not just in the boundary axis, but also others.
                 * To fit into the assumed form, we sum up all terms along the other axes.
                 * When doing it for non-Cherenkov-free directions, it would be same as Yee derivative.
                 * So that case is covered already and we only need to specialize the Cherenkov-free direction.
                 *
                 * @tparam T_axis derivative axis equal to Charenkov-free direction in Lehe solver, 0 = x, 1 = y, 2 = z
                 */
                template<uint32_t T_axis>
                struct DerivativeCoefficients<maxwellSolver::lehe::DerivativeFunctor<T_axis, T_axis>, T_axis>
                {
                    //! Number of coefficients = number of neighbors used in each side
                    static constexpr uint32_t size = 2;

                    //! Normalized coefficient values, actual coefficient = values / step
                    float_X values[size];

                    //! Instantiate and initialize derivative coefficients
                    HDINLINE DerivativeCoefficients()
                    {
                        float_64 const stepRatio = cellSize[T_axis] / (SPEED_OF_LIGHT * DELTA_T);
                        float_64 const coeff = stepRatio
                            * math::sin(pmacc::math::Pi<float_64>::halfValue * float_64(SPEED_OF_LIGHT)
                                        * float_64(DELTA_T) / float_64(cellSize[T_axis]));
                        auto const delta = static_cast<float_X>(0.25 * (1.0 - coeff * coeff));
                        /* values[0] = alpha + 2 * betaDir1 + 2 * betaDir2 = 1 - 3 * delta.
                         * Note that same as for 4th-order FDTD, values[0] + 3 * values[1] = 1.
                         */
                        values[0] = 1.0_X - 3.0_X * delta;
                        values[1] = delta;
                    }
                };

            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
