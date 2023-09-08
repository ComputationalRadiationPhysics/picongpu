/* Copyright 2015-2023 Rene Widera, Axel Huebl, Sergei Bastrakov, Klaus Steiniger
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

                    /** Creates an array with assignment values assuming that the position of the particle is on
                     * support.
                     *
                     * @tparam T_size Number of elements within the resulting array. Only the first 5 elements will be
                     * filled with valid values.
                     * @param  x particle position relative to the assignment cell range [-0.5;0.5)
                     * @return array with evaluated shape values
                     */
                    template<uint32_t T_size>
                    HDINLINE auto shapeArray(float_X const x) const
                    {
                        static_assert(T_size >= 5);
                        pmacc::memory::Array<float_X, T_size> shapeValues;
                        // grid points [-2;2]
                        shapeValues[0] = ff_3rd_radius(math::abs(-2._X - x));
                        shapeValues[1] = ff_2nd_radius(math::abs(-1._X - x));
                        // note: math::abs(0 - x) == math::abs(x)
                        shapeValues[2] = ff_1st_radius(math::abs(x));
                        shapeValues[4] = ff_3rd_radius(2._X - x);
                        // equal to ff_2nd_radius(1._X - x); but less compute intensive
                        shapeValues[3] = 1.0_X - (shapeValues[0] + shapeValues[1] + shapeValues[2] + shapeValues[4]);
                        return shapeValues;
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

                struct ChargeAssignment : public detail::PCS
                {
                    // lowest valid grid offsets
                    static constexpr int begin = -2;
                    // highest valid grid offsets
                    static constexpr int end = 3;

                    HDINLINE float_X operator()(float_X const x) const
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

                    /** Creates an array with assignment values.
                     *
                     * @param pos particle position relative to the assignment cell range [-0.5;1.5)
                     * @param isOutOfRange True if pos in range [-0.5;1.5)
                     * @return Array with precomputed assignment values.
                     */
                    HDINLINE auto shapeArray(float_X const xx, bool const isOutOfRange) const
                    {
                        float_X x = isOutOfRange ? xx - 1.0_X : xx;

                        auto shapeValues = detail::PCS::shapeArray<support + 1>(x);

                        // Update value so that a particle can be out of range without using lmem/local memory on GPUs
                        // because of dynamic indexing into an array located in registers.
                        shapeValues[5] = isOutOfRange ? shapeValues[4] : 0.0_X;
                        shapeValues[4] = isOutOfRange ? shapeValues[3] : shapeValues[4];
                        shapeValues[3] = isOutOfRange ? shapeValues[2] : shapeValues[3];
                        shapeValues[2] = isOutOfRange ? shapeValues[1] : shapeValues[2];
                        shapeValues[1] = isOutOfRange ? shapeValues[0] : shapeValues[1];
                        shapeValues[0] = isOutOfRange ? 0.0_X : shapeValues[0];

                        return shapeValues;
                    }
                };

                struct ChargeAssignmentOnSupport : public detail::PCS
                {
                    // lowest valid grid offsets
                    static constexpr int begin = -2;
                    // highest valid grid offsets
                    static constexpr int end = 2;

                    HDINLINE float_X operator()(float_X const x) const
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

                    /** Creates an array with assignment values.
                     *
                     * @param pos particle position relative to the assignment cell range [-0.5;0.5)
                     * @param isOutOfRange must be false, input will be ignored because the particle shape is always on
                     *                     support.
                     * @return Array with precomputed assignment values.
                     */
                    HDINLINE auto shapeArray(float_X const x, [[maybe_unused]] bool const isOutOfRange) const
                    {
                        return detail::PCS::shapeArray<support>(x);
                    }
                };
            };

        } // namespace shapes
    } // namespace particles
} // namespace picongpu
