/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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
                struct NGP
                {
                    /** Support of the assignment function in cells
                     *
                     * Specifies width of the area where the function can be non-zero.
                     * Is the same for all directions
                     */
                    static constexpr uint32_t support = 1;

                    /** Creates an array with assignment values assuming that the position of the particle is on
                     * support.
                     *
                     * @tparam T_size Number of elements within the resulting array. Only the first three elements will
                     * be filled with valid values.
                     * @param  x particle position relative to the assignment cell range [-0.5;0.5)
                     * @return array with evaluated shape values
                     */
                    template<uint32_t T_size>
                    HDINLINE auto shapeArray(float_X const x) const
                    {
                        static_assert(T_size >= 1);
                        pmacc::memory::Array<float_X, T_size> shapeValues;
                        // grid points [0;0]
                        shapeValues[0] = 1.0_X;
                        return shapeValues;
                    }
                };

            } // namespace detail

            /** Nearest grid point particle shape
             *
             * Cloud density form: delta function
             * Assignment function: zero order B-spline
             */
            struct NGP
            {
                //! Order of the assignment function spline
                static constexpr uint32_t assignmentFunctionOrder = detail::NGP::support - 1u;

                struct ChargeAssignment : public detail::NGP
                {
                    // lowest valid grid offsets
                    static constexpr int begin = 0;
                    // highest valid grid offsets
                    static constexpr int end = 1;

                    HDINLINE float_X operator()(float_X const x) const
                    {
                        /*       -
                         *       |  1               if -1/2<=x<1/2
                         * W(x)=<|
                         *       |  0               otherwise
                         *       -
                         */

                        bool const below_half = -0.5_X <= x && x < 0.5_X;

                        return float_X(below_half);
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

                        auto shapeValues = detail::NGP::shapeArray<support + 1>(x);

                        // Update value so that a particle can be out of range without using lmem/local memory on GPUs
                        // because of dynamic indexing into an array located in registers.
                        shapeValues[1] = isOutOfRange ? shapeValues[0] : 0.0_X;
                        shapeValues[0] = isOutOfRange ? 0.0_X : shapeValues[0];

                        return shapeValues;
                    }
                };

                struct ChargeAssignmentOnSupport : public detail::NGP
                {
                    // lowest valid grid offsets
                    static constexpr int begin = 0;
                    // highest valid grid offsets
                    static constexpr int end = 0;

                    /** form factor of this particle shape.
                     * @param x has to be within [-support/2, support/2)
                     */
                    HDINLINE float_X operator()(float_X const) const
                    {
                        /*
                         * W(x)=1
                         */
                        return 1.0_X;
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
                        return detail::NGP::shapeArray<support>(x);
                    }
                };
            };

        } // namespace shapes
    } // namespace particles
} // namespace picongpu
