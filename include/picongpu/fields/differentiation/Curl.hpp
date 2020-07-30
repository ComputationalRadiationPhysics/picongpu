/* Copyright 2013-2020 Axel Huebl, Heiko Burau, Rene Widera, Sergei Bastrakov
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

#include "picongpu/fields/differentiation/Curl.def"
#include "picongpu/fields/differentiation/Derivative.hpp"
#include "picongpu/traits/GetMargin.hpp"


namespace picongpu
{
namespace fields
{
namespace differentiation
{

    /** Functor to compute field curl at the given point
     *
     * @tparam T_Derivative derivative tag (not functor), defines the
     *                      finite-difference scheme for partial derivatives
     */
    template< typename T_Derivative >
    struct Curl
    {
        //! Derivative tag
        using Derivative = T_Derivative;

        //! Lower margin
        using LowerMargin = typename GetLowerMargin< Derivative >::type;

        //! Upper margin
        using UpperMargin = typename GetUpperMargin< Derivative >::type;

        //! Create curl functor
        HDINLINE Curl():
            xDerivativeFunctor(
                makeDerivativeFunctor<
                    Derivative,
                    0
                >()
            ),
            yDerivativeFunctor(
                makeDerivativeFunctor<
                    Derivative,
                    1
                >()
            ),
            zDerivativeFunctor(
                makeDerivativeFunctor<
                    Derivative,
                    2
                >()
            )
        {
        }

        /** Return curl value at the given point
         *
         * @tparam T_DataBox data box type with field data
         */
        template< typename T_DataBox >
        HDINLINE typename T_DataBox::ValueType operator()(
            T_DataBox const & data ) const
        {
            auto const dFdx = xDerivative( data );
            auto const dFdy = yDerivative( data );
            auto const dFdz = zDerivative( data );
            return float3_X{
                dFdy.z() - dFdz.y(),
                dFdz.x() - dFdx.z(),
                dFdx.y() - dFdy.x()
            };
        }

        /** Return x derivative value at the given point
         *
         * @tparam T_DataBox data box type with field data
         */
        template< typename T_DataBox >
        HDINLINE typename T_DataBox::ValueType xDerivative(
            T_DataBox const & data ) const
        {
            return xDerivativeFunctor( data );
        }

        /** Return y derivative value at the given point
         *
         * @tparam T_DataBox data box type with field data
         */
        template< typename T_DataBox >
        HDINLINE typename T_DataBox::ValueType yDerivative(
            T_DataBox const & data ) const
        {
            return yDerivativeFunctor( data );
        }

        /** Return z derivative value at the given point
         *
         * @tparam T_DataBox data box type with field data
         */
        template< typename T_DataBox >
        HDINLINE typename T_DataBox::ValueType zDerivative(
            T_DataBox const & data ) const
        {
            return zDerivativeFunctor( data );
        }

    private:

        using XDerivativeFunctor = decltype(
            makeDerivativeFunctor<
                Derivative,
                0
            >()
        );
        using YDerivativeFunctor = decltype(
            makeDerivativeFunctor<
                Derivative,
                1
            >()
        );
        using ZDerivativeFunctor = decltype(
            makeDerivativeFunctor<
                Derivative,
                2
            >()
        );

        XDerivativeFunctor const xDerivativeFunctor;
        YDerivativeFunctor const yDerivativeFunctor;
        ZDerivativeFunctor const zDerivativeFunctor;

    };

} // namespace differentiation
} // namespace fields
} // namespace picongpu
