/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
                struct Counter
                {
                    /** Support of the assignment function in cells
                     *
                     * Specifies width of the area where the function can be non-zero.
                     * Is the same for all directions.
                     * Note that the support is actually 1, but this shape is used only for
                     * certain operations and not as the main simulation shape, and so for
                     * enabling more generic implementations is set to zero.
                     */
                    static constexpr uint32_t support = 0;
                };

            } // namespace detail

            /** Version of nearest grid point particle shape used for counting particles
             *
             * Not to be used as a general particle shape in a simulation
             *
             * Cloud density form: delta function, shifted by half cell
             * Assignment function: zero order B-spline, shifted by half cell
             */
            struct Counter
            {
                /** Order of the assignment function spline
                 *
                 * Note that here the detail::Counter::support - 1u expression would
                 * not work, as the support of that shape is artificially set to 0
                 */
                static constexpr uint32_t assignmentFunctionOrder = 0u;

                struct ChargeAssignment : public detail::Counter
                {
                    HDINLINE float_X operator()(float_X const x) const
                    {
                        /*       -
                         *       | -1               if -1<x<=0
                         * W(x)=<|
                         *       |  0               otherwise
                         *       -
                         */

                        bool const in_cell = -1.0_X < x && x <= 0.0_X;

                        return float_X(in_cell);
                    }
                };

                struct ChargeAssignmentOnSupport : public detail::Counter
                {
                    /** form factor of this particle shape.
                     * @param x has to be within [0, 1)
                     */
                    HDINLINE float_X operator()(float_X const x) const
                    {
                        bool const in_cell = 0.0_X <= x && x < 1.0_X;

                        return float_X(in_cell);
                    }
                };
            };

        } // namespace shapes
    } // namespace particles
} // namespace picongpu
