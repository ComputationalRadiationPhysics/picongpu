/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Remi Lehe,
 *                     Sergei Bastrakov
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

#include "picongpu/fields/differentiation/Derivative.hpp"
#include "picongpu/fields/differentiation/ForwardDerivative.hpp"
#include "picongpu/fields/differentiation/Traits.hpp"
#include "picongpu/fields/MaxwellSolver/Lehe/Derivative.def"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/meta/accessors/Identity.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/types.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            namespace lehe
            {
                /** Functor for derivative used in the Lehe solver
                 *
                 * Implements eq. (6) in R. Lehe et al
                 *     Phys. Rev. ST Accel. Beams 16, 021301 (2013)
                 * This derivative can only be applied for the E field.
                 *
                 * @tparam T_cherenkovFreeDirection direction to remove numerical Cherenkov
                 *                                  radiation in, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
                 */
                template<uint32_t T_cherenkovFreeDirection, uint32_t T_direction>
                struct DerivativeFunctor;

                /** Functor for derivative along the Cherenkov free direction
                 *
                 * Implements eq. (6) in R. Lehe et al
                 *     Phys. Rev. ST Accel. Beams 16, 021301 (2013)
                 *
                 * @tparam T_direction Cherenkov free direction and derivative direction,
                 *                     0 = x, 1 = y, 2 = z
                 */
                template<uint32_t T_direction>
                struct DerivativeFunctor<T_direction, T_direction>
                {
                private:
                    //! Internally used derivative functor
                    using InternalDerivativeFunctor
                        = differentiation::DerivativeFunctor<differentiation::Forward, T_direction>;

                public:
                    /** Lower margin: we move by 1 along each direction and
                     *  apply InternalDerivativeFunctor, add those up
                     */
                    using LowerMargin = typename pmacc::math::CT::add<
                        typename pmacc::math::CT::make_Int<simDim, 1>::type,
                        typename GetLowerMargin<InternalDerivativeFunctor>::type>::type;

                    /** Upper margin: we move by 1 along each direction and
                     *  effectively apply InternalDerivativeFunctor (for T_direction not
                     *  literally, but structurally), add those up
                     */
                    using UpperMargin = typename pmacc::math::CT::add<
                        typename pmacc::math::CT::make_Int<simDim, 1>::type,
                        typename GetUpperMargin<InternalDerivativeFunctor>::type>::type;

                    //! Create a functor
                    HDINLINE DerivativeFunctor()
                    {
                        // differentiate along dir0; dir1 and dir2 are the other two directions
                        constexpr uint32_t dir0 = T_direction;
                        constexpr uint32_t dir1 = (dir0 + 1) % 3;
                        constexpr uint32_t dir2 = (dir0 + 2) % 3;

                        float_64 const stepRatio = cellSize[dir0] / (SPEED_OF_LIGHT * DELTA_T);
                        float_64 const coeff = stepRatio
                            * math::sin(pmacc::math::Pi<float_64>::halfValue * float_64(SPEED_OF_LIGHT)
                                        * float_64(DELTA_T) / float_64(cellSize[dir0]));
                        delta = static_cast<float_X>(0.25 * (1.0 - coeff * coeff));
                        // for 2D the betas corresponding to z are 0
                        float_64 const stepRatio1 = dir1 < simDim ? cellSize[dir0] / cellSize[dir1] : 0.0;
                        float_64 const stepRatio2 = dir2 < simDim ? cellSize[dir0] / cellSize[dir2] : 0.0;
                        float_64 const betaDir1 = 0.125 * stepRatio1 * stepRatio1;
                        float_64 const betaDir2 = 0.125 * stepRatio2 * stepRatio2;
                        alpha = static_cast<float_X>(1.0 - 2.0 * betaDir1 - 2.0 * betaDir2 - 3.0 * delta);
                    }

                    /** Return derivative value at the given point
                     *
                     * @tparam T_DataBox data box type with field data
                     * @param data position in the data box to compute derivative at
                     */
                    template<typename T_DataBox>
                    HDINLINE typename T_DataBox::ValueType operator()(T_DataBox const& data) const
                    {
                        // differentiate along dir0; dir1 and dir2 are the other two directions
                        constexpr uint32_t dir0 = T_direction;
                        constexpr uint32_t dir1 = (dir0 + 1) % 3;
                        constexpr uint32_t dir2 = (dir0 + 2) % 3;

                        // cellSize is not constexpr currently, so make an own constexpr array
                        constexpr float_X step[3] = {CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH};

                        /* beta_xy and beta_xz from eq. (11), generic for any T_direction;
                         * for 2D the betas corresponding to z are 0
                         */
                        constexpr float_X stepRatio1 = dir1 < simDim ? step[dir0] / step[dir1] : 0.0_X;
                        constexpr float_X stepRatio2 = dir2 < simDim ? step[dir0] / step[dir2] : 0.0_X;
                        constexpr float_X betaDir1 = 0.125_X * stepRatio1 * stepRatio1;
                        constexpr float_X betaDir2 = 0.125_X * stepRatio2 * stepRatio2;

                        // finite-difference expression from eq. (6), generic for any T_direction
                        using Index = pmacc::DataSpace<simDim>;
                        auto const secondUpperIndexDir0 = 2 * pmacc::math::basisVector<Index, dir0>();
                        auto const lowerIndexDir0 = -pmacc::math::basisVector<Index, dir0>();
                        auto const upperNeighborDir1 = pmacc::math::basisVector<Index, dir1>();
                        auto const upperNeighborDir2 = pmacc::math::basisVector<Index, dir2>();
                        InternalDerivativeFunctor forwardDerivative
                            = differentiation::makeDerivativeFunctor<differentiation::Forward, T_direction>();
                        return alpha * forwardDerivative(data)
                            + betaDir1 * forwardDerivative(data.shift(upperNeighborDir1))
                            + betaDir1 * forwardDerivative(data.shift(-upperNeighborDir1))
                            + betaDir2 * forwardDerivative(data.shift(upperNeighborDir2))
                            + betaDir2 * forwardDerivative(data.shift(-upperNeighborDir2))
                            + delta * (data(secondUpperIndexDir0) - data(lowerIndexDir0)) / step[T_direction];
                    }

                private:
                    //! alpha_x from eq. (7), generic for any T_direction
                    float_X alpha;

                    //! delta_x0 from eq. (10), generic for any T_direction
                    float_X delta;
                };

                /** Functor for derivative not along the Cherenkov free direction
                 *
                 * Implements eq. (6) in R. Lehe et al
                 *     Phys. Rev. ST Accel. Beams 16, 021301 (2013)
                 * Implementation is separated as a few terms vanish in this case
                 *
                 * @tparam T_cherenkovFreeDirection direction to remove numerical Cherenkov
                 *                                  radiation in, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction to take derivative in, not equal to
                 *                     T_cherenkovFreeDirection, 0 = x, 1 = y, 2 = z
                 */
                template<uint32_t T_cherenkovFreeDirection, uint32_t T_direction>
                struct DerivativeFunctor
                {
                    PMACC_CASSERT_MSG(
                        _lehe_solver_cherenkov_free_direction_z_is_not_supported_for_2d,
                        T_cherenkovFreeDirection < simDim);

                    PMACC_CASSERT_MSG(
                        _internal_error_wrong_lehe_derivative_functor_specialization,
                        T_cherenkovFreeDirection != T_direction);

                private:
                    //! Internally used derivative functor
                    using InternalDerivativeFunctor
                        = differentiation::DerivativeFunctor<differentiation::Forward, T_direction>;

                public:
                    /** Lower margin: we move by 1 along T_cherenkovFreeDirection and
                     *  apply InternalDerivativeFunctor, add those up
                     */
                    using LowerMargin = typename pmacc::math::CT::add<
                        typename pmacc::math::CT::make_BasisVector<simDim, T_cherenkovFreeDirection, int>::type,
                        typename GetLowerMargin<InternalDerivativeFunctor>::type>::type;

                    /** Upper margin: we move by 1 along T_cherenkovFreeDirection and
                     *  apply InternalDerivativeFunctor, add those up
                     */
                    using UpperMargin = typename pmacc::math::CT::add<
                        typename pmacc::math::CT::make_BasisVector<simDim, T_cherenkovFreeDirection, int>::type,
                        typename GetUpperMargin<InternalDerivativeFunctor>::type>::type;

                    /** Return derivative value at the given point
                     *
                     * @tparam T_DataBox data box type with field data
                     * @param data position in the data box to compute derivative at
                     */
                    template<typename T_DataBox>
                    HDINLINE typename T_DataBox::ValueType operator()(T_DataBox const& data) const
                    {
                        /* To obtain the following scheme, consider eq. (6) for x direction
                         * being Cherenkov-free and taking derivatives along y, z.
                         * Then in eq. (11) delta_y = delta_z = 0, beta_yz = beta_zy = 0,
                         * so only 3 terms are left in the derivative expression.
                         * It is implemented generically for any T_cherenkovFreeDirection
                         * and T_direction that are not equal to one another
                         */
                        constexpr float_X beta = 0.125_X;
                        constexpr float_X alpha = 1.0_X - 2.0_X * beta;
                        InternalDerivativeFunctor forwardDerivative
                            = differentiation::makeDerivativeFunctor<differentiation::Forward, T_direction>();
                        auto const upperNeighbor
                            = pmacc::math::basisVector<pmacc::DataSpace<simDim>, T_cherenkovFreeDirection>();
                        return alpha * forwardDerivative(data) + beta * forwardDerivative(data.shift(upperNeighbor))
                            + beta * forwardDerivative(data.shift(-upperNeighbor));
                    }
                };

            } // namespace lehe
        } // namespace maxwellSolver

        namespace differentiation
        {
            namespace traits
            {
                /** Functor type trait specialization for the Lehe solver derivative derivative
                 *
                 * @tparam T_cherenkovFreeDirection direction to remove numerical Cherenkov
                 *                                  radiation in, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
                 */
                template<uint32_t T_cherenkovFreeDirection, uint32_t T_direction>
                struct DerivativeFunctor<maxwellSolver::lehe::Derivative<T_cherenkovFreeDirection>, T_direction>
                    : pmacc::meta::accessors::Identity<
                          maxwellSolver::lehe::DerivativeFunctor<T_cherenkovFreeDirection, T_direction>>
                {
                };

            } // namespace traits
        } // namespace differentiation
    } // namespace fields
} // namespace picongpu
