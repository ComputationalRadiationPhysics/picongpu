/* Copyright 2018-2021 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/pluginSystem/TimeSlice.hpp"

#include <vector>


namespace pmacc
{
    namespace pluginSystem
    {
        /** check if a given step is within an interval list
         *
         * @param seqTimeSlices vector with time intervals
         * @param timeStep simulation time step to check
         * @return true if step is included in the interval list else false
         */
        HINLINE bool containsStep(std::vector<pluginSystem::TimeSlice> const& seqTimeSlices, uint32_t const timeStep)
        {
            for(auto const& timeSlice : seqTimeSlices)
            {
                if(timeStep >= timeSlice.values[0] && timeStep <= timeSlice.values[1])
                {
                    uint32_t const timeRelativeToStart = timeStep - timeSlice.values[0];
                    if(timeRelativeToStart % timeSlice.values[2] == 0)
                        return true;
                }
            }
            return false;
        }
    } // namespace pluginSystem
} // namespace pmacc
