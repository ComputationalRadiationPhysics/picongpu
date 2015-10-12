/**
 * Copyright 2013-2014 Heiko Burau, Rene Widera
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

#include "simulation_defines.hpp"

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
    BOOST_STATIC_CONSTEXPR int support = 3;


    HDINLINE static float_X ff_1st_radius(const float_X x)
    {
        /*
         * W(x)=3/4 - x^2
         */
        const float_X square_x = x*x;
        return float_X(0.75) - square_x;
    }

    HDINLINE static float_X ff_2nd_radius(const float_X x)
    {
        /*
         * W(x)=1/2*(3/2 - |x|)^2
         */
        const float_X tmp = (float_X(3.0 / 2.0) - x);
        const float_X square_tmp = tmp*tmp;
        return float_X(0.5) * square_tmp;
    }
};

} //namespace shared_TSC

struct TSC : public shared_TSC::TSC
{
    typedef  picongpu::particles::shapes::CIC CloudShape;

    struct ChargeAssignment : public shared_TSC::TSC
    {

        HDINLINE float_X operator()(const float_X x)
        {
            /*       -
             *       |  3/4 - x^2                  if |x|<1/2
             * W(x)=<|  1/2*(3/2 - |x|)^2          if 1/2<=|x|<3/2
             *       |  0                          otherwise
             *       -
             */
            float_X abs_x = algorithms::math::abs(x);

            const bool below_05 = (abs_x < float_X(0.5));
            const float_X fbelow_05 = float_X(below_05);

            return fbelow_05 * ff_1st_radius(abs_x) +
                float_X(abs_x < float_X(1.5) && !below_05) * ff_2nd_radius(abs_x);

        }
    };

    struct ChargeAssignmentOnSupport : public shared_TSC::TSC
    {

        /** form factor of this particle shape.
         * \param x has to be within [-support/2, support/2]
         */
        HDINLINE float_X operator()(const float_X x)
        {
            /*       -
             *       |  3/4 - x^2                  if |x|<1/2
             * W(x)=<|
             *       |  1/2*(3/2 - |x|)^2          if 1/2<=|x|<3/2
             *       -
             */
            float_X abs_x = algorithms::math::abs(x);

            const bool below_05 = (abs_x < float_X(0.5));
            const float_X fbelow_05 = float_X(below_05);

            return fbelow_05 * ff_1st_radius(abs_x) +
                float_X(!below_05) * ff_2nd_radius(abs_x);
        }

    };

};

} // namespace shapes
} // namespace partciles
} // namespace picongpu
