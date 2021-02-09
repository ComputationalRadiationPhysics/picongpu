/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Sergei Bastrakov
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
                struct NGP
                {
                    /** Support of the assignment function in cells
                     *
                     * Specifies width of the area where the function can be non-zero.
                     * Is the same for all directions
                     */
                    static constexpr uint32_t support = 1;
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
                    HDINLINE float_X operator()(float_X const x)
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
                };

                struct ChargeAssignmentOnSupport : public detail::NGP
                {
                    /** form factor of this particle shape.
                     * \param x has to be within [-support/2, support/2)
                     */
                    HDINLINE float_X operator()(float_X const)
                    {
                        /*
                         * W(x)=1
                         */
                        return 1.0_X;
                    }
                };
            };

        } // namespace shapes
    } // namespace particles
} // namespace picongpu
