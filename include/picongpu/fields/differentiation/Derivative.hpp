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
#include "picongpu/fields/differentiation/BackwardDerivative.hpp"
#include "picongpu/fields/differentiation/Derivative.def"
#include "picongpu/fields/differentiation/ForwardDerivative.hpp"
#include "picongpu/fields/differentiation/Traits.hpp"
#include "picongpu/fields/differentiation/ZeroDerivative.hpp"

#include <cstdint>


namespace picongpu
{
    namespace fields
    {
        namespace differentiation
        {
            /** Interface of field derivative functors created by makeDerivativeFunctor()
             *
             * In addition to operator(), the functor must be copyable and assignable.
             */
            struct DerivativeFunctorConcept
            {
                /** Return derivative value at the given point
                 *
                 * @tparam T_DataBox data box type with field data
                 * @param data position in the data box to compute derivative at
                 */
                template<typename T_DataBox>
                HDINLINE typename T_DataBox::ValueType operator()(T_DataBox const& data) const;
            };

            /** Type of derivative functor for the given derivative tag and direction
             *
             * Derivative tag defines the scheme and is used for configuration, while
             * the functor actually computes the derivatives along the given direction.
             *
             * @tparam T_Derivative derivative tag, defines the finite-difference scheme
             * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
             */
            template<typename T_Derivative, uint32_t T_direction>
            using DerivativeFunctor = typename traits::DerivativeFunctor<T_Derivative, T_direction>::type;

            /** Create a functor to compute field derivative along the given direction
             *
             * In case T_direction is >= simDim, returns the zero derivative functor
             *
             * @tparam T_Derivative derivative tag, defines the finite-difference scheme
             * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
             */
            template<typename T_Derivative, uint32_t T_direction>
            HDINLINE auto makeDerivativeFunctor()
            {
                return traits::MakeDerivativeFunctor<T_Derivative, T_direction>{}();
            }

        } // namespace differentiation
    } // namespace fields
} // namespace picongpu
