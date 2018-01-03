/* Copyright 2013-2018 Heiko Burau, Rene Widera
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



    HDINLINE static float_X ff_1st_radius(const float_X x)
    {
        /*
         * W(x)=1/6*(4 - 6*x^2 + 3*|x|^3)
         */
        const float_X square_x = x*x;
        const float_X triple_x = square_x*x;
        return float_X(1.0 / 6.0)*(float_X(4.0) - float_X(6.0) * square_x + float_X(3.0) * triple_x);
    }

    HDINLINE static float_X ff_2nd_radius(const float_X x)
    {
        /*
         * W(x)=1/6*(2 - |x|)^3
         */
        const float_X tmp = (float_X(2.0) - x);
        const float_X triple_tmp = tmp * tmp * tmp;
        return float_X(1.0 / 6.0) * triple_tmp;
    }
};

} //namespace shared_PCS
struct PCS : public shared_PCS::PCS
{
    using CloudShape = picongpu::particles::shapes::TSC;

    struct ChargeAssignment : public shared_PCS::PCS
    {

        HDINLINE float_X operator()(const float_X x)
        {
            /*       -
             *       |  1/6*(4 - 6*x^2 + 3*|x|^3)   if 0<=|x|<1
             * W(x)=<|  1/6*(2 - |x|)^3             if 1<=|x|<2
             *       |  0                           otherwise
             *       -
             */
            float_X abs_x = algorithms::math::abs(x);

            const bool below_1 = abs_x < float_X(1.0);
            const bool below_2 = abs_x < float_X(2.0);

            const float_X rad1 = ff_1st_radius(abs_x);
            const float_X rad2 = ff_2nd_radius(abs_x);

            float_X result(0.0);
            if(below_1)
                result = rad1;
            else if(below_2)
                result = rad2;

            return result;
        }
    };

    struct ChargeAssignmentOnSupport : public shared_PCS::PCS
    {

        HDINLINE float_X operator()(const float_X x)
        {
            /*       -
             *       |  1/6*(4 - 6*x^2 + 3*|x|^3)   if 0<=|x|<1
             * W(x)=<|
             *       |  1/6*(2 - |x|)^3             if 1<=|x|<2
             *       -
             */
            float_X abs_x = algorithms::math::abs(x);

            const bool below_1 = abs_x < float_X(1.0);
            const float_X rad1 = ff_1st_radius(abs_x);
            const float_X rad2 = ff_2nd_radius(abs_x);

            float_X result = rad2;
            if(below_1)
                result = rad1;

            return result;

            /* Semantix:
            if (abs_x < float_X(1.0))
                return ff_1st_radius(abs_x);
            return ff_2nd_radius(abs_x);
             */
        }

    };

};

} // namespace shapes
} //namespace particles
} //namespace picongpu
