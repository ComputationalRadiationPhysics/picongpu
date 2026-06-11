/*
 * SPDX-FileCopyrightText: Rene Widera, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/differentiation/BackwardDerivative.hpp"
#include "picongpu/fields/differentiation/Derivative.def"
#include "picongpu/fields/differentiation/ForwardDerivative.hpp"
#include "picongpu/fields/differentiation/Traits.hpp"
#include "picongpu/fields/differentiation/ZeroDerivative.hpp"

#include <cstdint>

namespace picongpu::fields::differentiation
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

    namespace detail
    {
        //! functor to compute field derivative along the given direction
        template<typename T_Derivative, uint32_t T_direction, bool T_isSpatialDimension>
        struct MakeDerivativeFunctor
        {
            using type = DerivativeFunctor<T_Derivative, T_direction>;
        };

        //! defines the zero derivative functor
        template<typename T_Derivative, uint32_t T_direction>
        struct MakeDerivativeFunctor<T_Derivative, T_direction, false>
        {
            using type = DerivativeFunctor<Zero, T_direction>;
        };

        // clang-format off
        //! Create a functor type to compute field derivative along the given direction
        template<typename T_Derivative, uint32_t T_direction>
        using MakeDerivativeFunctor_t = typename MakeDerivativeFunctor<
              T_Derivative,
              T_direction,
              T_direction < simDim
        >::type;
        // clang-format on
    } // namespace detail

    /** Create a functor to compute field derivative along the given direction
     *
     * @tparam T_Derivative derivative tag, defines the finite-difference scheme
     * @tparam T_direction direction to take derivative in, 0 = x, 1 = y, 2 = z
     */
    template<typename T_Derivative, uint32_t T_direction>
    HDINLINE auto makeDerivativeFunctor()
    {
        return detail::MakeDerivativeFunctor_t<T_Derivative, T_direction>{};
    }
} // namespace picongpu::fields::differentiation
