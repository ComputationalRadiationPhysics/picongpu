/* Copyright 2015-2018 Rene Widera
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

namespace shared_P4S
{

struct P4S
{
    static constexpr int support = 5;

    HDINLINE static float_X ff_1st_radius(const float_X x)
    {
        /*
         * W(x)= 115/192 - 5/8 * x^2 + 1/4 * x^4
         *     = 115/192 + x^2 * (-5/8 + 1/4 * x^2)
         */
        const float_X square_x = x * x;
        return float_X(115. / 192.)
            + square_x
            * (
               float_X(-5. / 8.)
               + float_X(1.0 / 4.0) * square_x
               );
    }

    HDINLINE static float_X ff_2nd_radius(const float_X x)
    {
        /*
         * W(x)= 1/96 * (55 + 20 * x - 120 * x^2 + 80 * x^3 - 16 * x^4)
         *     = 1/96 * (55 + 4 * x * (5 - 2 * x * (15 + 2 * x * (-5 + x))))
         */
        return float_X(1. / 96.)*
            (
             float_X(55.) + float_X(4.) * x
             * (float_X(5.) - float_X(2.) * x
                * (float_X(15.) + float_X(2.) * x
                   * (float_X(-5.) + x)
                   )
                )
             );
    }

    HDINLINE static float_X ff_3rd_radius(const float_X x)
    {
        /*
         * W(x)=1/384 * (5 - 2*x)^4
         */
        const float_X tmp = (float_X(5.) - float_X(2.) * x);
        const float_X square_tmp = tmp * tmp;
        const float_X biquadratic_tmp = square_tmp*square_tmp;

        return float_X(1. / 384.) * biquadratic_tmp;
    }
};

} //namespace shared_P4S

/** particle assignment shape `piecewise biquadratic spline`
 */
struct P4S : public shared_P4S::P4S
{
    using CloudShape = picongpu::particles::shapes::PCS;

    struct ChargeAssignmentOnSupport : public shared_P4S::P4S
    {

        HDINLINE float_X operator()(const float_X x)
        {
            /*       -
             *       |  115/192 + x^2 * (-5/8 + 1/4 * x^2)                          if -1/2 < x < 1/2
             * W(x)=<|
             *       |  1/96 * (55 + 4 * x * (5 - 2 * x * (15 + 2 * x * (-5 + x)))) if 1/2 <= |x| < 3/2
             *       |
             *       |  1/384 * (5 - 2 * x)^4                                       if 3/2 <= |x| < 5/2
             *       -
             */
            float_X abs_x = algorithms::math::abs(x);

            const bool below_2nd_radius = abs_x < float_X(1.5);
            const bool below_1st_radius = abs_x < float_X(0.5);

            const float_X rad1 = ff_1st_radius(abs_x);
            const float_X rad2 = ff_2nd_radius(abs_x);
            const float_X rad3 = ff_3rd_radius(abs_x);

            float_X result = rad3;
            if(below_1st_radius)
                result = rad1;
            else if(below_2nd_radius)
                result = rad2;

            return result;
        }

    };

    struct ChargeAssignment : public shared_P4S::P4S
    {

        HDINLINE float_X operator()(const float_X x)
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
            float_X abs_x = algorithms::math::abs(x);

            const bool below_max = abs_x < float_X(2.5);

            const float_X onSupport = ChargeAssignmentOnSupport()(abs_x);

            float_X result(0.0);
            if(below_max)
                result = onSupport;

            return result;
        }
    };
};

} // namespace shapes
} //namespace particles
} //namespace picongpu
