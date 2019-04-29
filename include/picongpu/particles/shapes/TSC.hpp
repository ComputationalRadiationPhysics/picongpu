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

namespace shared_TSC
{

struct TSC
{
    /**
     * width of the support of this form_factor. This is the area where the function
     * is non-zero.
     */
    static constexpr int support = 3;


    HDINLINE static float_X ff_1st_radius( float_X const x )
    {
        /*
         * W(x)=3/4 - x^2
         */
        float_X const square_x = x * x;
        return 0.75_X - square_x;
    }

    HDINLINE static float_X ff_2nd_radius( float_X const x )
    {
        /*
         * W(x)=1/2*(3/2 - |x|)^2
         */
        float_X const tmp = 3.0_X / 2.0_X - x;
        float_X const square_tmp = tmp * tmp;
        return 0.5_X * square_tmp;
    }
};

} //namespace shared_TSC

struct TSC : public shared_TSC::TSC
{
    using CloudShape = picongpu::particles::shapes::CIC;

    struct ChargeAssignment : public shared_TSC::TSC
    {

        HDINLINE float_X operator()( float_X const x )
        {
            /*       -
             *       |  3/4 - x^2                  if |x|<1/2
             * W(x)=<|  1/2*(3/2 - |x|)^2          if 1/2<=|x|<3/2
             *       |  0                          otherwise
             *       -
             */
            float_X const abs_x = algorithms::math::abs( x );

            bool const below_05 = abs_x < 0.5_X;
            bool const below_1_5 = abs_x < 1.5_X;

            float_X const rad1 = ff_1st_radius( abs_x );
            float_X const rad2 = ff_2nd_radius( abs_x );

            float_X result( 0.0 );
            if( below_05 )
                result = rad1;
            else if( below_1_5 )
                result = rad2;

            return result;

        }
    };

    struct ChargeAssignmentOnSupport : public shared_TSC::TSC
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
            float_X const abs_x = algorithms::math::abs( x );

            bool const below_05 = abs_x < 0.5_X;

            float_X const rad1 = ff_1st_radius( abs_x );
            float_X const rad2 = ff_2nd_radius( abs_x );

            float_X result = rad2;
            if( below_05 )
                result = rad1;

            return result;
        }

    };

};

} // namespace shapes
} // namespace partciles
} // namespace picongpu
