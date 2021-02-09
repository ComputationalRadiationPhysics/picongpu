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

#include "pmacc/types.hpp"
#include "pmacc/pluginSystem/TimeSlice.hpp"
#include "pmacc/misc/splitString.hpp"
#include "pmacc/verify.hpp"

#include <string>
#include <regex>
#include <vector>
#include <iostream>
#include <algorithm>
#include <array>


namespace pmacc
{
    namespace pluginSystem
    {
        namespace detail
        {
            /** check if string contains only digits
             *
             * @param str string to check
             * @return true if str contains only digits else false
             */
            HINLINE bool is_number(std::string const& str)
            {
                return std::all_of(str.begin(), str.end(), ::isdigit);
            }
        } // namespace detail

        /** create a TimeSlice out of an string
         *
         * Parse a comma separated list of time slices and creates a vector of TimeSlices.
         * TimeSlice Syntax:
         *   - `start:stop:period`
         *   - a number ``N is equal to `::N`
         */
        HINLINE std::vector<TimeSlice> toTimeSlice(std::string const& str)
        {
            std::vector<TimeSlice> result;
            auto const seqOfSlices = misc::splitString(str, ",");
            for(auto const& slice : seqOfSlices)
            {
                auto const sliceComponents = misc::splitString(slice, ":");
                PMACC_VERIFY_MSG(
                    !sliceComponents.empty(),
                    std::string("time slice without a defined element is not allowed") + str);

                // id of the component
                size_t n = 0;
                bool const hasOnlyPeriod = sliceComponents.size() == 1u;
                TimeSlice timeSlice;
                for(auto& component : sliceComponents)
                {
                    // be sure that component it is a number or empty
                    PMACC_VERIFY_MSG(
                        component.empty() || detail::is_number(component),
                        std::string("value") + component + " in " + str + "is not a number");

                    timeSlice.setValue(hasOnlyPeriod ? 2 : n, component);
                    n++;
                }
                result.push_back(timeSlice);
            }
            return result;
        }
    } // namespace pluginSystem
} // namespace pmacc
