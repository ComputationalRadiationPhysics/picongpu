/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Axel Huebl, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
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
