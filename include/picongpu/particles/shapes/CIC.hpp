/* Copyright 2013-2023 Heiko Burau, Rene Widera, Axel Huebl, Sergei Bastrakov
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

#include "picongpu/defines.hpp"

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
                struct CIC
                {
                    /** Support of the assignment function in cells
                     *
                     * Specifies width of the area where the function can be non-zero.
                     * Is the same for all directions
                     */
                    static constexpr uint32_t support = 2;

                    /** Creates an array with assignment values assuming that the position of the particle is on
                     * support.
                     *
                     * @tparam T_size Number of elements within the resulting array. Only the first two elements will
                     * be filled with valid values.
                     * @param  x particle position relative to the assignment cell range [0.0;1.0)
                     * @return array with evaluated shape values
                     */
                    template<uint32_t T_size>
                    HDINLINE auto shapeArray(float_X const x) const
                    {
                        static_assert(T_size >= 2);
                        pmacc::memory::Array<float_X, T_size> shapeValues;
                        // grid points [0;1]
                        // x is always postive therefore abs(x) is not required
                        shapeValues[0] = 1.0_X - x;
                        // x == 1 - 1 - abs(x)
                        shapeValues[1] = x;

                        return shapeValues;
                    }
                };

            } // namespace detail

            /** Cloud-in-cell particle shape
             *
             * Cloud density form: piecewise constant
             * Assignment function: first order B-spline
             */
            struct CIC
            {
                //! Order of the assignment function spline
                static constexpr uint32_t assignmentFunctionOrder = detail::CIC::support - 1u;

                struct ChargeAssignment : public detail::CIC
                {
                    // lowest valid grid offsets
                    static constexpr int begin = 0;
                    // highest valid grid offsets
                    static constexpr int end = 2;

                    HDINLINE float_X operator()(float_X const x) const
                    {
                        /*       -
                         *       |  1-|x|           if |x|<1
                         * W(x)=<|
                         *       |  0               otherwise
                         *       -
                         */
                        float_X const abs_x = math::abs(x);

                        bool const below_1 = abs_x < 1.0_X;
                        float_X const onSupport = 1.0_X - abs_x;

                        float_X result(0.0);
                        if(below_1)
                            result = onSupport;

                        return result;
                    }

                    /** Creates an array with assignment values.
                     *
                     * @param pos particle position relative to the assignment cell range [0.0;2.0)
                     * @param isOutOfRange True if pos in range [1.0;2.0)
                     * @return Array with precomputed assignment values.
                     */
                    HDINLINE auto shapeArray(float_X const pos, bool const isOutOfRange) const
                    {
                        float_X x = isOutOfRange ? pos - 1.0_X : pos;

                        auto shapeValues = detail::CIC::shapeArray<support + 1>(x);

                        // Update value so that a particle can be out of range without using lmem/local memory on GPUs
                        // because of dynamic indexing into an array located in registers.
                        shapeValues[2] = isOutOfRange ? shapeValues[1] : 0.0_X;
                        shapeValues[1] = isOutOfRange ? shapeValues[0] : shapeValues[1];
                        shapeValues[0] = isOutOfRange ? 0.0_X : shapeValues[0];

                        return shapeValues;
                    }
                };

                struct ChargeAssignmentOnSupport : public detail::CIC
                {
                    // lowest valid grid offsets
                    static constexpr int begin = 0;
                    // highest valid grid offsets
                    static constexpr int end = 1;

                    /** form factor of this particle shape.
                     * @param x has to be within [-support/2, support/2]
                     */
                    HDINLINE float_X operator()(float_X const x) const
                    {
                        /*
                         * W(x)=1-|x|
                         */
                        return 1.0_X - math::abs(x);
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
                        return detail::CIC::shapeArray<support>(x);
                    }
                };
            };

        } // namespace shapes
    } // namespace particles
} // namespace picongpu
