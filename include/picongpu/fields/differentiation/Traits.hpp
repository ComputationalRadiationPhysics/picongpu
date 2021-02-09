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

#include <cstdint>


namespace picongpu
{
    namespace fields
    {
        namespace differentiation
        {
            namespace traits
            {
                /** Type trait for derivative functor for the given derivative tag and
                 *  direction, accessible as ::type
                 *
                 * Has to be specialized for each derivative tag.
                 *
                 * @tparam T_Derivative derivative tag, defines the finite-difference scheme
                 * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
                 */
                template<typename T_Derivative, uint32_t T_direction>
                struct DerivativeFunctor;

                /** Factory for functors to compute field derivative along the given direction
                 *
                 * In case T_direction is >= simDim, returns the zero derivative functor.
                 * Does not need to be specialized when DerivativeFunctor is specialized.
                 *
                 * @tparam T_Derivative derivative tag, defines the finite-difference scheme
                 * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
                 * @tparam T_isLesserThanDim flag to decide between normal and zero derivative
                 */
                template<typename T_Derivative, uint32_t T_direction, bool T_isLesserThanDim = (T_direction < simDim)>
                struct MakeDerivativeFunctor
                {
                    using Functor = typename DerivativeFunctor<T_Derivative, T_direction>::type;

                    //! Return a functor
                    HDINLINE Functor operator()() const
                    {
                        return Functor{};
                    }
                };

                /** Factory for functors to compute field derivative along the given direction
                 *
                 * Implementation for T_direction >= simDim, always returns zero derivative
                 *
                 * @tparam T_Derivative derivative tag, defines the finite-difference scheme
                 * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
                 */
                template<typename T_Derivative, uint32_t T_direction>
                struct MakeDerivativeFunctor<T_Derivative, T_direction, false>
                {
                    using ZeroFunctor = typename DerivativeFunctor<Zero, T_direction>::type;

                    //! Return a zero functor
                    HDINLINE ZeroFunctor operator()() const
                    {
                        return ZeroFunctor{};
                    }
                };

            } // namespace traits
        } // namespace differentiation
    } // namespace fields
} // namespace picongpu
