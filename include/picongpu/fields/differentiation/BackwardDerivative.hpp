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
            /** Functor for backward difference derivative along the given direction
             *
             * Computes (current - lower) / step, previously called DifferenceToLower.
             *
             * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
             */
            template<uint32_t T_direction>
            struct BackwardDerivativeFunctor
            {
                //! Lower margin
                using LowerMargin = typename pmacc::math::CT::make_BasisVector<simDim, T_direction, int>::type;

                //! Upper margin
                using UpperMargin = typename pmacc::math::CT::make_Int<simDim, 0>::type;

                /** Return derivative value at the given point
                 *
                 * @tparam T_DataBox data box type with field data
                 * @param data position in the data box to compute derivative at
                 */
                template<typename T_DataBox>
                HDINLINE typename T_DataBox::ValueType operator()(T_DataBox const& data) const
                {
                    using Index = pmacc::DataSpace<simDim>;
                    auto const lowerIndex = -pmacc::math::basisVector<Index, T_direction>();
                    return (data(Index{}) - data(lowerIndex)) / sim.pic.getCellSize()[T_direction];
                }
            };

            namespace traits
            {
                /** Functor type trait specialization for backward derivative
                 *
                 * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
                 */
                template<uint32_t T_direction>
                struct DerivativeFunctor<Backward, T_direction>
                    : pmacc::meta::accessors::Identity<BackwardDerivativeFunctor<T_direction>>
                {
                };

            } // namespace traits
        } // namespace differentiation
    } // namespace fields
} // namespace picongpu
