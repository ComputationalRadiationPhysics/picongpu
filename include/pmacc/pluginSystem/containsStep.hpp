/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/pluginSystem/Slice.hpp"

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
        inline bool containsStep(std::vector<pluginSystem::Slice> const& seqTimeSlices, uint32_t const timeStep)
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
