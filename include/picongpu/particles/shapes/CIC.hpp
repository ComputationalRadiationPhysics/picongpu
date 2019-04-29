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
namespace shared_CIC
{

struct CIC
{
    /**
     * width of the support of this form_factor. This is the area where the function
     * is non-zero.
     */
    static constexpr int support = 2;
};

}//namespace shared_CIC

struct CIC : public shared_CIC::CIC
{
    using CloudShape = picongpu::particles::shapes::NGP;

    struct ChargeAssignment : public shared_CIC::CIC
    {

        HDINLINE float_X operator()( float_X const x )
        {
            /*       -
             *       |  1-|x|           if |x|<1
             * W(x)=<|
             *       |  0               otherwise
             *       -
             */
            float_X const abs_x = algorithms::math::abs( x );

            bool const below_1 = abs_x < 1.0_X;
            float_X const onSupport = 1.0_X - abs_x;

            float_X result( 0.0 );
            if( below_1 )
                result = onSupport;

            return result;
        }
    };

    struct ChargeAssignmentOnSupport : public shared_CIC::CIC
    {

        /** form factor of this particle shape.
         * \param x has to be within [-support/2, support/2]
         */
        HDINLINE float_X operator()( float_X const x )
        {
            /*
             * W(x)=1-|x|
             */
            return 1.0_X - algorithms::math::abs( x );
        }

    };

};

} // namespace shapes
} // namespace particles
} // namespace picongpu
