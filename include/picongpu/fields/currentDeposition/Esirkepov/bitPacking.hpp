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

#include "picongpu/fields/currentDeposition/Esirkepov/Line.hpp"

#include <cstdint>


namespace picongpu
{
    namespace currentSolver
    {
        /** Helper base class for Esirkepov to manage bit packing operations to encode the status of the start and end
         * particle to optimize the algorithm.
         * Bit-packing for the status is required to keep the register footprint low.
         */
        namespace bitpacking
        {
            enum class Status
            {
                /** Mark that a particle trajectory is crossing the assignment cell border.
                 *
                 * For performance reason it is required that LEAVE_CELL is represented by the first bit. This allow
                 * that we can use getValue() to querry the integer value. It is used to optimize the loop contition in
                 * the Esirkepov implemenation.
                 */
                LEAVE_CELL = 1,
                //! Mark that the start particle is located in the base assignment cell.
                START_PARTICLE_IN_ASSIGNMENT_CELL = 2,
                //! Mark that the end particle is located in the base assignment cell.
                END_PARTICLE_IN_ASSIGNMENT_CELL = 4
            };

            /*! Set status in the bitmask
             *
             * @param packedValue status bit mask
             * @param status status type
             * @param condition true sets the bit, false keeps current status.
             */
            DINLINE void set(int& packedValue, Status const status, bool condition)
            {
                packedValue |= condition ? static_cast<int>(status) : 0;
            }

            /*! Query the integer value of a status type
             *
             * The value is depending on the bit represented by the status.
             *
             * @param packedValue status bit mask
             * @param status status type
             * @return integer value of the status, zero means status is not active.
             */
            DINLINE int getValue(int packedValue, Status status)
            {
                return (packedValue & static_cast<int>(status));
            }

            /*! Test the status for a given type
             *
             * @param packedValue status bit mask
             * @param status status type
             * @return true if status is set else false
             */
            DINLINE bool test(int packedValue, Status status)
            {
                return getValue(packedValue, status) != 0;
            }

        }; // namespace bitpacking

    } // namespace currentSolver
} // namespace picongpu
