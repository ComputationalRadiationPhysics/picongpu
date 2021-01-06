/* Copyright 2013-2021 Heiko Burau, Rene Widera, Axel Huebl, Sergei Bastrakov
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
                struct CIC
                {
                    /** Support of the assignment function in cells
                     *
                     * Specifies width of the area where the function can be non-zero.
                     * Is the same for all directions
                     */
                    static constexpr uint32_t support = 2;
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
                    HDINLINE float_X operator()(float_X const x)
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
                };

                struct ChargeAssignmentOnSupport : public detail::CIC
                {
                    /** form factor of this particle shape.
                     * \param x has to be within [-support/2, support/2]
                     */
                    HDINLINE float_X operator()(float_X const x)
                    {
                        /*
                         * W(x)=1-|x|
                         */
                        return 1.0_X - math::abs(x);
                    }
                };
            };

        } // namespace shapes
    } // namespace particles
} // namespace picongpu
