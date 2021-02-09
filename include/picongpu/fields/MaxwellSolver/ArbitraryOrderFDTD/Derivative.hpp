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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/fields/MaxwellSolver/ArbitraryOrderFDTD/Derivative.def"
#include "picongpu/fields/differentiation/Traits.hpp"
#include <picongpu/fields/MaxwellSolver/ArbitraryOrderFDTD/Weights.hpp>

#include <pmacc/math/Vector.hpp>
#include <pmacc/meta/accessors/Identity.hpp>

#include <cstdint>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            namespace aoFDTD
            {
                namespace detail
                {
                    /** Abstraction of the arbitrary-order finite-difference time domain
                     *  derivative functor.
                     *
                     * @tparam T_lowerNeighbors Number of neighbors required in negative
                     *                          direction to calculate field derivative
                     *
                     *
                     * @tparam T_upperNeighbors Number of neighbors required in positive
                     *                          direction to calculate field derivative
                     *
                     * @tparam T_neighbors Number of neighbors used to calculate
                     *                     the derivative from finite differences.
                     *                     Order of derivative approximation is
                     *                     2 * T_neighbors
                     *
                     * @tparam T_direction Direction to take derivative in, 0 = x, 1 = y, 2 = z
                     */
                    template<
                        uint32_t T_lowerNeighbors,
                        uint32_t T_upperNeighbors,
                        uint32_t T_neighbors,
                        uint32_t T_direction>
                    struct GeneralAofdtdDerivative
                    {
                        //! Lower margin
                        using LowerMargin = typename pmacc::math::CT::mul<
                            typename pmacc::math::CT::make_Int<simDim, T_lowerNeighbors>::type,
                            typename pmacc::math::CT::make_BasisVector<simDim, T_direction, int>::type>::type;

                        //! Upper margin
                        using UpperMargin = typename pmacc::math::CT::mul<
                            typename pmacc::math::CT::make_Int<simDim, T_upperNeighbors>::type,
                            typename pmacc::math::CT::make_BasisVector<simDim, T_direction, int>::type>::type;

                        /** Return derivative value at the given point
                         *
                         * @tparam T_DataBox data box type with field data
                         * @param data position in the data box to compute derivative at
                         */
                        template<typename T_DataBox>
                        HDINLINE typename T_DataBox::ValueType operator()(T_DataBox const& data) const
                        {
                            // Define shorthand type to access DataBox
                            using IndexType = pmacc::DataSpace<simDim>;

                            // Define indice vectors for data access
                            auto lowerIndex = IndexType{}; // Vector initialized with zeros
                            auto upperIndex = IndexType{};

                            // lowerIndex: 0 if ( Forward ) else -1
                            lowerIndex[T_direction]
                                = static_cast<int32_t>(T_upperNeighbors) - static_cast<int32_t>(T_neighbors);
                            // upperIndex: 1 if ( Forward ) else 0
                            upperIndex[T_direction]
                                = static_cast<int32_t>(T_neighbors) - static_cast<int32_t>(T_lowerNeighbors);

                            AOFDTDWeights<T_neighbors> const weights{};

                            // shortest distance finite difference as initial value
                            auto finiteDifference = weights[0] * (data(upperIndex) - data(lowerIndex));

                            // Compute next finite differences according to order
                            for(uint32_t l = 1u; l < T_neighbors; ++l)
                            {
                                lowerIndex[T_direction] -= 1;
                                upperIndex[T_direction] += 1;

                                finiteDifference += weights[l] * (data(upperIndex) - data(lowerIndex));
                            }

                            return finiteDifference / cellSize[T_direction];
                        }
                    };
                } // namespace detail


                /**@{*/
                /** Functors for forward and backward derivative along the given direction used in ArbitraryOrderFDTD
                 * solver
                 *
                 * Compute an approximation of the derivative of a field f by a finite difference of
                 * order 2 * T_neighbors, where T_neighbors is the number of neighbors
                 * used to calculate the finite difference.
                 *
                 * This finite difference approximations for the forward and backward derivative are computed on a
                 * staggered grid. That is, the forward derivative will be known at a position i+1/2, if the field f is
                 * known at 2 * T_neighbors grid nodes i - T_neighbors + 1, i - T_neighbors + 2, ..., i + T_neighbors.
                 * The backward derivative will be known at a position i-1/2, if the field f is known
                 * at 2 * T_neighbors grid nodes i - T_neighbors, i - T_neighbors + 1, ..., i + T_neighbors - 1.
                 *
                 * The finite difference calculation can be expressed as a sum of finite differences where the
                 * distance of field components used in individual finite differences computations increases, e.g.
                 *     D_x f(i+1/2) = sum_{l=0}^{T_neighbors-1} g_l^{2T_neighbors} * ( f(i+1+l) - f(i-l) ) / dx,
                 * for the forward derivative and where D_x is the derivative operator along x, dx the grid spacing
                 * along x, and g_l^{2T_neighbors} weightings for the finite differences of different distance l from
                 * the point i of computation.
                 *
                 * @tparam T_neighbors Number of neighbors used to calculate
                 *                     the derivative from finite differences.
                 *                     Order of derivative approximation is
                 *                     2 * T_neighbors
                 *
                 * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
                 */
                template<uint32_t T_neighbors, uint32_t T_direction>
                using ForwardDerivativeFunctor
                    = detail::GeneralAofdtdDerivative<T_neighbors - 1, T_neighbors, T_neighbors, T_direction>;


                template<uint32_t T_neighbors, uint32_t T_direction>
                using BackwardDerivativeFunctor
                    = detail::GeneralAofdtdDerivative<T_neighbors, T_neighbors - 1, T_neighbors, T_direction>;
                /**@}*/

            } // namespace aoFDTD
        } // namespace maxwellSolver

        namespace differentiation
        {
            namespace traits
            {
                /**@{*/
                /** DerivativeFunctor type trait specialization for Forward and Backward derivative in
                 *  ArbitraryOrderFDTD solver
                 *
                 * @tparam T_neighbors Number of neighbors used to calculate
                 *                      the derivative from finite differences.
                 *                      Order of derivative approximation is
                 *                      2 * T_neighbors
                 *
                 * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
                 */
                template<uint32_t T_neighbors, uint32_t T_direction>
                struct DerivativeFunctor<maxwellSolver::aoFDTD::Forward<T_neighbors>, T_direction>
                    : pmacc::meta::accessors::Identity<
                          maxwellSolver::aoFDTD::ForwardDerivativeFunctor<T_neighbors, T_direction>>
                {
                };


                template<uint32_t T_neighbors, uint32_t T_direction>
                struct DerivativeFunctor<maxwellSolver::aoFDTD::Backward<T_neighbors>, T_direction>
                    : pmacc::meta::accessors::Identity<
                          maxwellSolver::aoFDTD::BackwardDerivativeFunctor<T_neighbors, T_direction>>
                {
                };
                /**@}*/

            } // namespace traits
        } // namespace differentiation
    } // namespace fields
} // namespace picongpu
