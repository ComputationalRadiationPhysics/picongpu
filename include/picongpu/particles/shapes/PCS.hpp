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

namespace shared_PCS
{
struct PCS
{
    static constexpr int support = 4;



    HDINLINE static float_X ff_1st_radius( float_X const x )
    {
        /*
         * W(x)=1/6*(4 - 6*x^2 + 3*|x|^3)
         */
        float_X const square_x = x * x;
        float_X const triple_x = square_x * x;
        return 1.0_X / 6.0_X * ( 4.0_X - 6.0_X * square_x + 3.0_X * triple_x );
    }

    HDINLINE static float_X ff_2nd_radius( float_X const x )
    {
        /*
         * W(x)=1/6*(2 - |x|)^3
         */
        float_X const tmp = 2.0_X - x;
        float_X const triple_tmp = tmp * tmp * tmp;
        return 1.0_X / 6.0_X * triple_tmp;
    }
};

} //namespace shared_PCS
struct PCS : public shared_PCS::PCS
{
    using CloudShape = picongpu::particles::shapes::TSC;

    struct ChargeAssignment : public shared_PCS::PCS
    {

        HDINLINE float_X operator()( float_X const x )
        {
            /*       -
             *       |  1/6*(4 - 6*x^2 + 3*|x|^3)   if 0<=|x|<1
             * W(x)=<|  1/6*(2 - |x|)^3             if 1<=|x|<2
             *       |  0                           otherwise
             *       -
             */
            float_X const abs_x = algorithms::math::abs( x );

            bool const below_1 = abs_x < 1.0_X;
            bool const below_2 = abs_x < 2.0_X;

            float_X const rad1 = ff_1st_radius( abs_x );
            float_X const rad2 = ff_2nd_radius( abs_x );

            float_X result( 0.0 );
            if( below_1 )
                result = rad1;
            else if( below_2 )
                result = rad2;

            return result;
        }
    };

    struct ChargeAssignmentOnSupport : public shared_PCS::PCS
    {

        HDINLINE float_X operator()( float_X const x )
        {
            /*       -
             *       |  1/6*(4 - 6*x^2 + 3*|x|^3)   if 0<=|x|<1
             * W(x)=<|
             *       |  1/6*(2 - |x|)^3             if 1<=|x|<2
             *       -
             */
            float_X const abs_x = algorithms::math::abs( x );

            bool const below_1 = abs_x < 1.0_X;
            float_X const rad1 = ff_1st_radius( abs_x );
            float_X const rad2 = ff_2nd_radius( abs_x );

            float_X result = rad2;
            if( below_1 )
                result = rad1;

            return result;

            /* Semantics:
            if( abs_x < 1.0_X )
                return ff_1st_radius( abs_x );
            return ff_2nd_radius( abs_x );
             */
        }

    };

};

} // namespace shapes
} // namespace particles
} // namespace picongpu
