/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Remi Lehe,
 *                     Sergei Bastrakov, Lennert Sprenger
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/fields/MaxwellSolver/CKC/Derivative.def"
#include "picongpu/fields/differentiation/Derivative.hpp"
#include "picongpu/fields/differentiation/ForwardDerivative.hpp"
#include "picongpu/fields/differentiation/Traits.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/meta/accessors/Identity.hpp>
#include <pmacc/types.hpp>

#include <algorithm>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            namespace ckc
            {
                /** Functor for derivative used in the CKC solver
                 *
                 * Implements the modified derivative from
                 *     Phys. Rev. ST Accel. Beams 16, 041303 (2013)
                 *   https://dx.doi.org/10.1103/PhysRevSTAB.16.041303
                 *
                 * This derivative can only be applied for the E field.
                 *
                 * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
                 */
                template<uint32_t T_direction>
                struct DerivativeFunctor
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

                    /** Return derivative value at the given point
                     *
                     * @tparam T_DataBox data box type with field data
                     * @param data position in the data box to compute derivative at
                     */
                    template<typename T_DataBox>
                    HDINLINE typename T_DataBox::ValueType operator()(T_DataBox const& data) const
                    {
                        constexpr uint32_t dir0 = T_direction;
                        constexpr uint32_t dir1 = (dir0 + 1) % 3;
                        constexpr uint32_t dir2 = (dir0 + 2) % 3;

                        constexpr float_64 step[3] = {CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH};
                        constexpr float_64 step2[3] = {step[0] * step[0], step[1] * step[1], step[2] * step[2]};

                        // delta: smallest step size
                        constexpr float_64 delta = std::min({CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH});

                        constexpr float_64 delta2 = delta * delta;

                        // r_i = delta^2 / (deltax_i)^2
                        constexpr float_64 r[3] = {delta2 / step2[0], delta2 / step2[1], delta2 / step2[2]};

                        constexpr float_64 r012 = r[0] * r[1] * r[2];
                        constexpr float_64 rprodsum = r[0] * r[1] + r[1] * r[2] + r[2] * r[0];

                        // equation (21)
                        constexpr float_64 betaFactor = 0.125 * (1.0 - r012 / rprodsum);
                        constexpr float_X betaDir1 = static_cast<float_X>(r[dir1] * betaFactor);
                        constexpr float_X betaDir2 = static_cast<float_X>(r[dir2] * betaFactor);

                        // equation (22)
                        constexpr float_X gammaDir12 = static_cast<float_X>(
                            r[dir1] * r[dir2] * (0.0625 - 0.125 * (r[dir1] * r[dir2] / rprodsum)));

                        // from the condition alpha + 2 * betaDir1 + 2 * betaDir2 + 4 * gammaDir12 = 1
                        constexpr float_X alpha
                            = static_cast<float_X>(1.0 - 2.0 * betaDir1 - 2.0 * betaDir2 - 4.0 * gammaDir12);

                        using Index = pmacc::DataSpace<simDim>;
                        auto const upperNeighborDir1 = pmacc::math::basisVector<Index, dir1>();
                        auto const upperNeighborDir2 = pmacc::math::basisVector<Index, dir2>();

                        InternalDerivativeFunctor forwardDerivative
                            = differentiation::makeDerivativeFunctor<differentiation::Forward, T_direction>();

                        // smoothing operator applied after the derivative
                        return alpha * forwardDerivative(data)
                            + betaDir1 * forwardDerivative(data.shift(upperNeighborDir1))
                            + betaDir1 * forwardDerivative(data.shift(-upperNeighborDir1))

                            + betaDir2 * forwardDerivative(data.shift(upperNeighborDir2))
                            + betaDir2 * forwardDerivative(data.shift(-upperNeighborDir2))

                            + gammaDir12 * forwardDerivative(data.shift(upperNeighborDir1 + upperNeighborDir2))
                            + gammaDir12 * forwardDerivative(data.shift(upperNeighborDir1 - upperNeighborDir2))
                            + gammaDir12 * forwardDerivative(data.shift(-upperNeighborDir1 + upperNeighborDir2))
                            + gammaDir12 * forwardDerivative(data.shift(-upperNeighborDir1 - upperNeighborDir2));
                    }
                };

            } // namespace ckc
        } // namespace maxwellSolver

        namespace differentiation
        {
            namespace traits
            {
                template<uint32_t T_direction>
                struct DerivativeFunctor<maxwellSolver::ckc::Derivative, T_direction>
                    : pmacc::meta::accessors::Identity<maxwellSolver::ckc::DerivativeFunctor<T_direction>>
                {
                };

            } // namespace traits
        } // namespace differentiation
    } // namespace fields
} // namespace picongpu
