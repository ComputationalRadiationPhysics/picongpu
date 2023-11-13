/* Copyright 2013-2023 Heiko Burau, Rene Widera, Axel Huebl, Sergei Bastrakov, Klaus Steiniger
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

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/Array.hpp>

#include <cstdint>


namespace picongpu
{
    namespace particles
    {
        namespace shapes
        {
            namespace detail
            {
                struct PQS
                {
                    /** Support of the assignment function in cells
                     *
                     * Specifies width of the area where the function can be non-zero.
                     * Is the same for all directions
                     */
                    static constexpr uint32_t support = 4;

                    HDINLINE static float_X ff_1st_radius(float_X const x)
                    {
                        /*
                         * W(x)=1/6*(4 - 6*x^2 + 3*|x|^3)
                         */
                        float_X const square_x = x * x;
                        float_X const triple_x = square_x * x;
                        return 1.0_X / 6.0_X * (4.0_X - 6.0_X * square_x + 3.0_X * triple_x);
                    }

                    HDINLINE static float_X ff_2nd_radius(float_X const x)
                    {
                        /*
                         * W(x)=1/6*(2 - |x|)^3
                         */
                        float_X const tmp = 2.0_X - x;
                        float_X const triple_tmp = tmp * tmp * tmp;
                        return 1.0_X / 6.0_X * triple_tmp;
                    }

                    /** Creates an array with assignment values assuming that the position of the particle is on
                     * support.
                     *
                     * @tparam T_size Number of elements within the resulting array. Only the first 4 elements will be
                     * filled with valid values.
                     * @param  x particle position relative to the assignment cell range [0.0;1.0)
                     * @return array with evaluated shape values
                     */
                    template<uint32_t T_size>
                    HDINLINE auto shapeArray(float_X const x) const
                    {
                        static_assert(T_size >= 4);
                        pmacc::memory::Array<float_X, T_size> shapeValues;
                        // grid points [-1;2]
                        shapeValues[0] = ff_2nd_radius(math::abs(-1._X - x));
                        // note: math::abs(0 - x) == math::abs(x)
                        shapeValues[1] = ff_1st_radius(x);
                        shapeValues[3] = ff_2nd_radius(2._X - x);
                        // equal to ff_1st_radius(1._X - x); but less compute intensive
                        shapeValues[2] = 1.0_X - (shapeValues[0] + shapeValues[1] + shapeValues[3]);
                        return shapeValues;
                    }
                };

            } // namespace detail

            /** Piecewise quadratic cloud particle shape
             *
             * Cloud density form: piecewise quadratic B-spline
             * Assignment function: piecewise cubic B-spline
             */
            struct PQS
            {
                //! Order of the assignment function spline
                static constexpr uint32_t assignmentFunctionOrder = detail::PQS::support - 1u;

                struct ChargeAssignment : public detail::PQS
                {
                    // lowest valid grid offsets
                    static constexpr int begin = -1;
                    // highest valid grid offsets
                    static constexpr int end = 3;

                    HDINLINE float_X operator()(float_X const x) const
                    {
                        /*       -
                         *       |  1/6*(4 - 6*x^2 + 3*|x|^3)   if 0<=|x|<1
                         * W(x)=<|  1/6*(2 - |x|)^3             if 1<=|x|<2
                         *       |  0                           otherwise
                         *       -
                         */
                        float_X const abs_x = math::abs(x);

                        bool const below_1 = abs_x < 1.0_X;
                        bool const below_2 = abs_x < 2.0_X;

                        float_X const rad1 = ff_1st_radius(abs_x);
                        float_X const rad2 = ff_2nd_radius(abs_x);

                        float_X result(0.0);
                        if(below_1)
                            result = rad1;
                        else if(below_2)
                            result = rad2;

                        return result;
                    }

                    /** Creates an array with assignment values.
                     *
                     * @param pos particle position relative to the assignment cell range [0.0;2.0)
                     * @param isOutOfRange True if pos in range [1.0;2.0)
                     * @return Array with precomputed assignment values.
                     */
                    HDINLINE auto shapeArray(float_X const xx, bool const isOutOfRange) const
                    {
                        float_X x = isOutOfRange ? xx - 1.0_X : xx;

                        auto shapeValues = detail::PQS::shapeArray<support + 1>(x);

                        // Update value so that a particle can be out of range without using lmem/local memory on GPUs
                        // because of dynamic indexing into an array located in registers.
                        shapeValues[4] = isOutOfRange ? shapeValues[3] : 0.0_X;
                        shapeValues[3] = isOutOfRange ? shapeValues[2] : shapeValues[3];
                        shapeValues[2] = isOutOfRange ? shapeValues[1] : shapeValues[2];
                        shapeValues[1] = isOutOfRange ? shapeValues[0] : shapeValues[1];
                        shapeValues[0] = isOutOfRange ? 0.0_X : shapeValues[0];

                        return shapeValues;
                    }
                };

                struct ChargeAssignmentOnSupport : public detail::PQS
                {
                    // lowest valid grid offsets
                    static constexpr int begin = -1;
                    // highest valid grid offsets
                    static constexpr int end = 2;

                    HDINLINE float_X operator()(float_X const x) const
                    {
                        /*       -
                         *       |  1/6*(4 - 6*x^2 + 3*|x|^3)   if 0<=|x|<1
                         * W(x)=<|
                         *       |  1/6*(2 - |x|)^3             if 1<=|x|<2
                         *       -
                         */
                        float_X const abs_x = math::abs(x);

                        bool const below_1 = abs_x < 1.0_X;
                        float_X const rad1 = ff_1st_radius(abs_x);
                        float_X const rad2 = ff_2nd_radius(abs_x);

                        float_X result = rad2;
                        if(below_1)
                            result = rad1;

                        return result;

                        /* Semantics:
                        if( abs_x < 1.0_X )
                            return ff_1st_radius( abs_x );
                        return ff_2nd_radius( abs_x );
                         */
                    }

                    /** Creates an array with assignment values.
                     *
                     * @param pos particle position relative to the assignment cell range [0.0;1.0)
                     * @param isOutOfRange must be false, input will be ignored because the particle shape is always on
                     *                     support.
                     * @return Array with precomputed assignment values.
                     */
                    HDINLINE auto shapeArray(float_X const x, [[maybe_unused]] bool const isOutOfRange) const
                    {
                        return detail::PQS::shapeArray<support>(x);
                    }
                };
            };

        } // namespace shapes
    } // namespace particles
} // namespace picongpu
