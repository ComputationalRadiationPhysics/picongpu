/* Copyright 2015-2021 Rene Widera, Axel Huebl, Sergei Bastrakov, Klaus Steiniger
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

#include <cstdint>


namespace picongpu
{
    namespace particles
    {
        namespace shapes
        {
            namespace detail
            {
                struct PCS
                {
                    /** Support of the assignment function in cells
                     *
                     * Specifies width of the area where the function can be non-zero.
                     * Is the same for all directions
                     */
                    static constexpr uint32_t support = 5;

                    HDINLINE static float_X ff_1st_radius(float_X const x)
                    {
                        /*
                         * W(x)= 115/192 - 5/8 * x^2 + 1/4 * x^4
                         *     = 115/192 + x^2 * (-5/8 + 1/4 * x^2)
                         */
                        float_X const square_x = x * x;
                        return 115._X / 192._X + square_x * (-5._X / 8._X + 1.0_X / 4.0_X * square_x);
                    }

                    HDINLINE static float_X ff_2nd_radius(float_X const x)
                    {
                        /*
                         * W(x)= 1/96 * (55 + 20 * x - 120 * x^2 + 80 * x^3 - 16 * x^4)
                         *     = 1/96 * (55 + 4 * x * (5 - 2 * x * (15 + 2 * x * (-5 + x))))
                         */
                        return 1._X / 96._X
                            * (55._X + 4._X * x * (5._X - 2._X * x * (15._X + 2._X * x * (-5._X + x))));
                    }

                    HDINLINE static float_X ff_3rd_radius(float_X const x)
                    {
                        /*
                         * W(x)=1/384 * (5 - 2*x)^4
                         */
                        float_X const tmp = 5._X - 2._X * x;
                        float_X const square_tmp = tmp * tmp;
                        float_X const biquadratic_tmp = square_tmp * square_tmp;

                        return 1._X / 384._X * biquadratic_tmp;
                    }
                };

            } // namespace detail

            /** Piecewise cubic cloud particle shape
             *
             * Cloud density form: piecewise cubic B-Spline
             * Assignment function: piecewise quartic B-spline
             */
            struct PCS
            {
                //! Order of the assignment function spline
                static constexpr uint32_t assignmentFunctionOrder = detail::PCS::support - 1u;

                struct ChargeAssignmentOnSupport : public detail::PCS
                {
                    HDINLINE float_X operator()(float_X const x)
                    {
                        /*       -
                         *       |  115/192 + x^2 * (-5/8 + 1/4 * x^2)                          if -1/2 < x < 1/2
                         * W(x)=<|
                         *       |  1/96 * (55 + 4 * x * (5 - 2 * x * (15 + 2 * x * (-5 + x)))) if 1/2 <= |x| < 3/2
                         *       |
                         *       |  1/384 * (5 - 2 * x)^4                                       if 3/2 <= |x| < 5/2
                         *       -
                         */
                        float_X const abs_x = math::abs(x);

                        bool const below_2nd_radius = abs_x < 1.5_X;
                        bool const below_1st_radius = abs_x < 0.5_X;

                        float_X const rad1 = ff_1st_radius(abs_x);
                        float_X const rad2 = ff_2nd_radius(abs_x);
                        float_X const rad3 = ff_3rd_radius(abs_x);

                        float_X result = rad3;
                        if(below_1st_radius)
                            result = rad1;
                        else if(below_2nd_radius)
                            result = rad2;

                        return result;
                    }
                };

                struct ChargeAssignment : public detail::PCS
                {
                    HDINLINE float_X operator()(float_X const x)
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
                        float_X const abs_x = math::abs(x);

                        bool const below_max = abs_x < 2.5_X;

                        float_X const onSupport = ChargeAssignmentOnSupport()(abs_x);

                        float_X result(0.0);
                        if(below_max)
                            result = onSupport;

                        return result;
                    }
                };
            };

        } // namespace shapes
    } // namespace particles
} // namespace picongpu
