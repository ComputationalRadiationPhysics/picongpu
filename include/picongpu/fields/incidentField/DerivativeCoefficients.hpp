/* Copyright 2021-2023 Sergei Bastrakov
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

#include <pmacc/math/Vector.hpp>

#include <cstdint>

#include <pmacc/math/vector/compile-time/Vector.hpp>


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
                 * The current implementation assumes Yee grid and central derivative operator are used.
                 * Thus, the derivative coefficients are antisymmetric along T_axis and symmetric along other axes.
                 *
                 * For example, for T_axis = 0 evaluating d/dx of incident field F at (integer or half-integer)
                 * Yee grid index (i0, j0, k0) follows the expression:
                 * dF/dx (i0, j0, k0) = (1.0 / stepX) *
                 * sum(value(i, |j|, |k|) * (F(i0 + (i + 1/2), j0 + j, k0 + k) - F(i0 - (i + 1/2), j0 + j, k0 + k));
                 * i, j, k are integers, i in [0, Size::x::value); j in [-(Size::y::value - 1), Size::y::value),
                 * k in [-(Size::z::value - 1), Size::z::value).
                 * For other axes it is defined similarly.
                 *
                 * The default implementation works for the classic 2nd order central derivative on the Yee grid.
                 * Each specialization must provide public Size and value members with the matching semantics.
                 * Each specialization object must be default-constructible on host and bitwise copyable.
                 *
                 * @note We do not make the difference between the forward and backward derivative.
                 * This is becase it is already part of index calculations in the host side of the incident field
                 * solver. Same (no difference) should hold for all specializations.
                 *
                 * @tparam T_DerivativeFunctor derivative functor type along the axis
                 * @tparam T_axis derivative axis, 0 = x, 1 = y, 2 = z
                 */
                template<typename T_DerivativeFunctor, uint32_t T_axis>
                struct DerivativeCoefficients
                {
                    /** Number of coefficients: 1 along each axis
                     *
                     * This size must be defined as if for 3d, regardless of simDim.
                     * Each component must be at least 1.
                     */
                    using Size = pmacc::math::CT::make_Int<3, 1>::type;

                    /** Normalized coefficient values, actual coefficient = value[...] / step[T_axis]
                     *
                     * This array size is controlled by type Size.
                     * The coefficients are for non-negative side terms (see equation above).
                     * For the negative side we assume antisymmetry along T_axis and symmetry along other axes.
                     */
                    float_X value[Size::x::value][Size::y::value][Size::z::value] = {{{1.0_X}}};
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
                    //! Number of coefficients: T_neighbors along T_axis, 1 along other axes
                    using Size = typename pmacc::math::CT::Assign<
                        pmacc::math::CT::make_Int<3, 1>::type::vector_type,
                        std::integral_constant<int, T_axis>,
                        std::integral_constant<int, T_neighbors>>::type;

                    //! Normalized coefficient values, actual coefficient = value[...] / step[T_axis]
                    float_X value[Size::x::value][Size::y::value][Size::z::value];

                    //! Instantiate and initialize derivative coefficients
                    DerivativeCoefficients()
                    {
                        // E.g. for 4th-order FDTD and d/dy, values[0][0][0] = 27/24, values[0][1][0] = -1/24
                        maxwellSolver::aoFDTD::AOFDTDWeights<T_neighbors> weights;
                        auto idx = pmacc::DataSpace<3>::create(0);
                        for(uint32_t i = 0; i < T_neighbors; i++)
                        {
                            idx[T_axis] = i;
                            value[idx.x()][idx.y()][idx.z()] = weights[i];
                        }
                    }
                };

                /** Specialization for the Lehe derivative along the Cherenkov-free direction
                 *
                 * @tparam T_axis derivative axis equal to Cherenkov-free direction in Lehe solver, 0 = x, 1 = y, 2 = z
                 */
                template<uint32_t T_axis>
                struct DerivativeCoefficients<maxwellSolver::lehe::DerivativeFunctor<T_axis, T_axis>, T_axis>
                {
                    /* Number of coefficients: 2 along each axis
                     * (on T_axis due to a wider stencil, on other axes due to the averaging)
                     */
                    using Size = typename pmacc::math::CT::make_Int<3, 2>::type;

                    //! Normalized coefficient values, actual coefficient = value[...] / step[T_axis]
                    float_X value[Size::x::value][Size::y::value][Size::z::value];

                    //! Instantiate and initialize derivative coefficients
                    DerivativeCoefficients()
                    {
                        constexpr uint32_t dir0 = T_axis;
                        constexpr uint32_t dir1 = (dir0 + 1) % 3;
                        constexpr uint32_t dir2 = (dir0 + 2) % 3;
                        float_64 const stepRatio = cellSize[dir0] / (SPEED_OF_LIGHT * DELTA_T);
                        float_64 const coeff = stepRatio
                            * math::sin(pmacc::math::Pi<float_64>::halfValue * float_64(SPEED_OF_LIGHT)
                                        * float_64(DELTA_T) / float_64(cellSize[dir0]));
                        auto const delta = static_cast<float_X>(0.25 * (1.0 - coeff * coeff));
                        // for 2D the betas corresponding to z are 0
                        float_64 const stepRatio1 = dir1 < simDim ? cellSize[dir0] / cellSize[dir1] : 0.0;
                        float_64 const stepRatio2 = dir2 < simDim ? cellSize[dir0] / cellSize[dir2] : 0.0;
                        float_64 const betaDir1 = 0.125 * stepRatio1 * stepRatio1;
                        float_64 const betaDir2 = 0.125 * stepRatio2 * stepRatio2;
                        auto const alpha = static_cast<float_X>(1.0 - 2.0 * betaDir1 - 2.0 * betaDir2 - 3.0 * delta);
                        // Initialize all with 0 as we won't set values to some elements later
                        for(uint32_t i = 0u; i < Size::x::value; i++)
                            for(uint32_t j = 0u; j < Size::y::value; j++)
                                for(uint32_t k = 0u; k < Size::z::value; k++)
                                    value[i][j][k] = 0.0_X;
                        value[0][0][0] = alpha;
                        auto idxDir0 = pmacc::DataSpace<3>::create(0);
                        idxDir0[dir0] = 1;
                        value[idxDir0[0]][idxDir0[1]][idxDir0[2]] = delta;
                        auto idxDir1 = pmacc::DataSpace<3>::create(0);
                        idxDir1[dir1] = 1;
                        value[idxDir1[0]][idxDir1[1]][idxDir1[2]] = betaDir1;
                        auto idxDir2 = pmacc::DataSpace<3>::create(0);
                        idxDir2[dir2] = 1;
                        value[idxDir2[0]][idxDir2[1]][idxDir2[2]] = betaDir2;
                    }
                };

                /** Specialization for the Lehe derivative not along the Cherenkov-free direction
                 *
                 * @tparam T_axis Cherenkov-free direction in Lehe solver, 0 = x, 1 = y, 2 = z
                 * @tparam T_axis derivative axis not equal to Cherenkov-free direction in Lehe solver, 0 = x, 1 = y, 2
                 * = z
                 */
                template<uint32_t T_cherenkovFreeDirection, uint32_t T_axis>
                struct DerivativeCoefficients<
                    maxwellSolver::lehe::DerivativeFunctor<T_cherenkovFreeDirection, T_axis>,
                    T_axis>
                {
                    //! Number of coefficients: 2 along T_cherenkovFreeDirection, 1 along other axes
                    using Size = typename pmacc::math::CT::Assign<
                        pmacc::math::CT::make_Int<3, 1>::type::vector_type,
                        std::integral_constant<int, T_cherenkovFreeDirection>,
                        std::integral_constant<int, 2>>::type;

                    //! Normalized coefficient values, actual coefficient = value[...] / step[T_axis]
                    float_X value[Size::x::value][Size::y::value][Size::z::value];

                    //! Instantiate and initialize derivative coefficients
                    DerivativeCoefficients()
                    {
                        constexpr float_X beta = 0.125_X;
                        constexpr float_X alpha = 1.0_X - 2.0_X * beta;
                        value[0][0][0] = alpha;
                        auto idx = pmacc::DataSpace<3>::create(0);
                        idx[T_cherenkovFreeDirection] = 1;
                        value[idx[0]][idx[1]][idx[2]] = beta;
                    }
                };

            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
