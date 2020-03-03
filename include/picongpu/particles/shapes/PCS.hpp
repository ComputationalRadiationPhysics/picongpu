/* Copyright 2013-2019 Heiko Burau, Rene Widera, Axel Huebl
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

namespace sharedPCS
{
struct PCS
{
    static constexpr int support = 4;



    HDINLINE static float_X ff1stRadius( float_X const x )
    {
        /*
         * W(x)=1/6*(4 - 6*x^2 + 3*|x|^3)
         */
        float_X const xSquared = x * x;
        float_X const xCube = xSquared * x;
        return 1.0_X / 6.0_X * ( 4.0_X - 6.0_X * xSquared + 3.0_X * xCube );
    }

    HDINLINE static float_X ff2ndRadius( float_X const x )
    {
        /*
         * W(x)=1/6*(2 - |x|)^3
         */
        float_X const tmp = 2.0_X - x;
        float_X const tmpCube = tmp * tmp * tmp;
        return 1.0_X / 6.0_X * tmpCube;
    }
};

} //namespace sharedPCS
struct PCS : public sharedPCS::PCS
{
    using CloudShape = picongpu::particles::shapes::TSC;

    struct ChargeAssignment : public sharedPCS::PCS
    {

        HDINLINE float_X operator()( float_X const x )
        {
            /*       -
             *       |  1/6*(4 - 6*x^2 + 3*|x|^3)   if 0<=|x|<1
             * W(x)=<|  1/6*(2 - |x|)^3             if 1<=|x|<2
             *       |  0                           otherwise
             *       -
             */
            float_X const xAbs = algorithms::math::abs( x );

            bool const isInSupport_1_0 = xAbs < 1.0_X;
            bool const isInSupport_2_0 = xAbs < 2.0_X;

            float_X const valueOnSupport_1_0 = ff1stRadius( xAbs );
            float_X const valueOnSupport_2_0 = ff2ndRadius( xAbs );

            float_X result( 0.0 );
            if( isInSupport_1_0 )
                result = valueOnSupport_1_0;
            else if( isInSupport_2_0 )
                result = valueOnSupport_2_0;

            return result;
        }
    };

    struct ChargeAssignmentOnSupport : public sharedPCS::PCS
    {

        HDINLINE float_X operator()( float_X const x )
        {
            /*       -
             *       |  1/6*(4 - 6*x^2 + 3*|x|^3)   if 0<=|x|<1
             * W(x)=<|
             *       |  1/6*(2 - |x|)^3             if 1<=|x|<2
             *       -
             */
            float_X const xAbs = algorithms::math::abs( x );

            bool const isInSupport_1_0 = xAbs < 1.0_X;
            float_X const valueOnSupport_1_0 = ff1stRadius( xAbs );
            float_X const valueOnSupport_2_0 = ff2ndRadius( xAbs );

            float_X result = valueOnSupport_2_0;
            if( isInSupport_1_0 )
                result = valueOnSupport_1_0;

            return result;

        }

    };

};

} // namespace shapes
} // namespace particles
} // namespace picongpu
