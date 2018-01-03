/* Copyright 2013-2018 Axel Huebl, Heiko Burau, Rene Widera
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

    namespace shared_Counter
    {

        struct Counter
        {
            /**
             * width of the support of this form_factor. This is the area where the function
             * is non-zero.
             */
            static constexpr int support = 0;
        };

    } // namespace shared_Counter

    struct Counter : public shared_Counter::Counter
    {

        struct ChargeAssignment : public shared_Counter::Counter
        {

            HDINLINE float_X operator()(const float_X x)
            {
                /*       -
                 *       | -1               if -1<x<=0
                 * W(x)=<|
                 *       |  0               otherwise
                 *       -
                 */

                const bool in_cell = ( float_X(-1.0) < x &&
                                                       x <= float_X(0.0) );

                return float_X(in_cell);
            }
        };

        struct ChargeAssignmentOnSupport : public shared_Counter::Counter
        {

            /** form factor of this particle shape.
             * \param x has to be within [-support/2, support/2)
             */
            HDINLINE float_X operator()(const float_X x)
            {
                const bool in_cell = ( float_X(0.0) <= x &&
                                                       x < float_X(1.0) );

                return float_X(in_cell);
            }

        };

    };

} // namespace shapes
} // namespace particles
} // namespace picongpu
