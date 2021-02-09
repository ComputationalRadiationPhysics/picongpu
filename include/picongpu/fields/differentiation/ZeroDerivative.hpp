/* Copyright 2013-2021 Heiko Burau, Rene Widera, Axel Huebl, Sergei Bastrakov
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
#include "picongpu/fields/differentiation/Derivative.def"
#include "picongpu/fields/differentiation/Traits.hpp"

#include <pmacc/math/Vector.hpp>
#include <pmacc/meta/accessors/Identity.hpp>

#include <cstdint>


namespace picongpu
{
    namespace fields
    {
        namespace differentiation
        {
            /** Functor for zero derivative along the given direction
             *
             * Always returns zero.
             *
             * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
             */
            template<uint32_t T_direction>
            struct ZeroDerivativeFunctor
            {
                //! Lower margin
                using LowerMargin = typename pmacc::math::CT::make_Int<simDim, 0>::type;

                //! Upper margin
                using UpperMargin = typename pmacc::math::CT::make_Int<simDim, 0>::type;

                /** Return zero
                 *
                 * @tparam T_DataBox data box type with field data
                 * @param data position in the data box to compute derivative at
                 */
                template<typename T_DataBox>
                HDINLINE typename T_DataBox::ValueType operator()(T_DataBox const& data) const
                {
                    return T_DataBox::ValueType::create(0.0_X);
                }
            };

            namespace traits
            {
                /** Functor type trait specialization for zero derivative
                 *
                 * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
                 */
                template<uint32_t T_direction>
                struct DerivativeFunctor<Zero, T_direction>
                    : pmacc::meta::accessors::Identity<ZeroDerivativeFunctor<T_direction>>
                {
                };

            } // namespace traits
        } // namespace differentiation
    } // namespace fields
} // namespace picongpu
