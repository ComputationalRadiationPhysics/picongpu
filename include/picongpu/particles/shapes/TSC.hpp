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

namespace sharedTSC
{

struct TSC
{
    /**
     * width of the support of this form_factor. This is the area where the function
     * is non-zero.
     */
    static constexpr int support = 3;


    HDINLINE static float_X ff1stRadius( float_X const x )
    {
        /*
         * W(x)=3/4 - x^2
         */
        float_X const xSquared = x * x;
        return 0.75_X - xSquared;
    }

    HDINLINE static float_X ff2ndRadius( float_X const x )
    {
        /*
         * W(x)=1/2*(3/2 - |x|)^2
         */
        float_X const tmp = 3.0_X / 2.0_X - x;
        float_X const tmpSquared = tmp * tmp;
        return 0.5_X * tmpSquared;
    }
};

} //namespace sharedTSC

struct TSC : public sharedTSC::TSC
{
    using CloudShape = picongpu::particles::shapes::CIC;

    struct ChargeAssignment : public sharedTSC::TSC
    {

        HDINLINE float_X operator()( float_X const x )
        {
            /*       -
             *       |  3/4 - x^2                  if |x|<1/2
             * W(x)=<|  1/2*(3/2 - |x|)^2          if 1/2<=|x|<3/2
             *       |  0                          otherwise
             *       -
             */
            float_X const xAbs = algorithms::math::abs( x );

            bool const isInSupport_0_5 = xAbs < 0.5_X;
            bool const isInSupport_1_5 = xAbs < 1.5_X;

            float_X const valueOnSupport_0_5 = ff1stRadius( xAbs );
            float_X const valueOnSupport_1_5 = ff2ndRadius( xAbs );

            float_X result( 0.0 );
            if( isInSupport_0_5 )
                result = valueOnSupport_0_5;
            else if( isInSupport_1_5 )
                result = valueOnSupport_1_5;

            return result;

        }
    };

    struct ChargeAssignmentOnSupport : public sharedTSC::TSC
    {

        /** form factor of this particle shape.
         * \param x has to be within [-support/2, support/2]
         */
        HDINLINE float_X operator()( float_X const x )
        {
            /*       -
             *       |  3/4 - x^2                  if |x|<1/2
             * W(x)=<|
             *       |  1/2*(3/2 - |x|)^2          if 1/2<=|x|<3/2
             *       -
             */
            float_X const xAbs = algorithms::math::abs( x );

            bool const isInSupport_0_5 = xAbs < 0.5_X;

            float_X const valueOnSupport_0_5 = ff1stRadius( xAbs );
            float_X const valueOnSupport_1_5 = ff2ndRadius( xAbs );

            float_X result = valueOnSupport_1_5;
            if( isInSupport_0_5 )
                result = valueOnSupport_0_5;

            return result;
        }

    };

};

} // namespace shapes
} // namespace partciles
} // namespace picongpu
