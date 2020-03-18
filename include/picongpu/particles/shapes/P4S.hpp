/* Copyright 2015-2019 Rene Widera, Axel Huebl
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

namespace picongpu
{
namespace particles
{
namespace shapes
{

namespace sharedP4S
{

struct P4S
{
    static constexpr int support = 5;

    HDINLINE static float_X ff1stRadius( float_X const x )
    {
        /*
         * W(x)= 115/192 - 5/8 * x^2 + 1/4 * x^4
         *     = 115/192 + x^2 * (-5/8 + 1/4 * x^2)
         */
        float_X const xSquared = x * x;
        return 115._X / 192._X + xSquared * (
            -5._X / 8._X +
            1.0_X / 4.0_X * xSquared
        );
    }

    HDINLINE static float_X ff2ndRadius( float_X const x )
    {
        /*
         * W(x)= 1/96 * (55 + 20 * x - 120 * x^2 + 80 * x^3 - 16 * x^4)
         *     = 1/96 * (55 + 4 * x * (5 - 2 * x * (15 + 2 * x * (-5 + x))))
         */
        return 1._X / 96._X * (
            55._X + 4._X * x * (
                5._X - 2._X * x * (
                    15._X + 2._X * x * (
                        -5._X + x
                    )
                )
            )
        );
    }

    HDINLINE static float_X ff3rdRadius( float_X const x )
    {
        /*
         * W(x)=1/384 * (5 - 2*x)^4
         */
        float_X const tmp = 5._X - 2._X * x;
        float_X const tmpSquared = tmp * tmp;
        float_X const quarticTmp = tmpSquared * tmpSquared;

        return 1._X / 384._X * quarticTmp;
    }
};

} //namespace sharedP4S

/** particle assignment shape `piecewise biquadratic spline`
 */
struct P4S : public sharedP4S::P4S
{
    using CloudShape = picongpu::particles::shapes::PCS;

    struct ChargeAssignmentOnSupport : public sharedP4S::P4S
    {

        HDINLINE float_X operator()( float_X const x )
        {
            /*       -
             *       |  115/192 + x^2 * (-5/8 + 1/4 * x^2)                          if -1/2 < x < 1/2
             * W(x)=<|
             *       |  1/96 * (55 + 4 * x * (5 - 2 * x * (15 + 2 * x * (-5 + x)))) if 1/2 <= |x| < 3/2
             *       |
             *       |  1/384 * (5 - 2 * x)^4                                       if 3/2 <= |x| < 5/2
             *       -
             */
            float_X const xAbs = algorithms::math::abs( x );

            bool const isInSupport_1_5 = xAbs < 1.5_X;
            bool const isInSupport_0_5 = xAbs < 0.5_X;

            float_X const valueOnSupport_1_5 = ff1stRadius( xAbs );
            float_X const valueOnSupport_0_5 = ff2ndRadius( xAbs );
            float_X const valueOnSupport_2_5 = ff3rdRadius( xAbs );

            float_X result = valueOnSupport_2_5;
            if( isInSupport_1_5 )
                result = valueOnSupport_1_5;
            else if( isInSupport_0_5 )
                result = valueOnSupport_0_5;

            return result;
        }

    };

    struct ChargeAssignment : public sharedP4S::P4S
    {

        HDINLINE float_X operator()( float_X const x )
        {

            /*       -
             *       |  115/192 + x^2 * (-5/8 + 1/4 * x^2)                          if -1/2 < x < 1/2
             * W(x)=<|
             *       |  1/96 * (55 + 4 * x * (5 - 2 * x * (15 + 2 * x * (-5 + x)))) if 1/2 <= |x| < 3/2
             *       |
             *       |  1/384 * (5 - 2*x)^4                                         if 3/2 <= |x| < 5/2
             *       |
             *       |  0                                                           otherwise
             *       -
             */
            float_X const xAbs = algorithms::math::abs( x );

            bool const isInSupport = xAbs < 2.5_X;

            float_X const valueOnSupport = ChargeAssignmentOnSupport()( xAbs );

            float_X result( 0.0 );
            if( isInSupport )
                result = valueOnSupport;

            return result;
        }
    };
};

} // namespace shapes
} //namespace particles
} //namespace picongpu
