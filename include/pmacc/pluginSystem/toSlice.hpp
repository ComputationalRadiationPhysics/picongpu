/* Copyright 2022-2023 Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/misc/splitString.hpp"
#include "pmacc/pluginSystem/Slice.hpp"
#include "pmacc/verify.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

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
            inline bool is_number(std::string const& str)
            {
                return std::all_of(str.begin(), str.end(), ::isdigit);
            }
        } // namespace detail

        /** create a TimeSlice out of an string
         *
         * Parse a comma separated list of time slices and creates a vector of slices.
         * time slice syntax:
         *   - `start:stop:period`
         *   - a number ``N is equal to `::N`
         *   - `:` is equal to `0:-1:1`
         *
         * If stop is `-1` there is no defined end.
         *
         * @param str Comma separated list of slices. Empty slices will be skipped.
         */
        inline std::vector<Slice> toTimeSlice(std::string const& str)
        {
            std::vector<Slice> result;
            auto const seqOfSlices = misc::splitString(str, ",");
            for(auto slice : seqOfSlices)
            {
                // skip empty slice strings
                if(slice.empty())
                    continue;
                else
                {
                    /* If the slice is not empty extent the input with one delimiter to get an empty string in cases
                     * where the slice ends with a delimiter without any other following characters.
                     */
                    slice += ":";
                }
                auto const sliceComponents = misc::splitString(slice, ":");
                PMACC_VERIFY_MSG(
                    !sliceComponents.empty(),
                    std::string("time slice without a defined element is not allowed") + str);

                // id of the component
                size_t n = 0;
                bool const hasOnlyPeriod = sliceComponents.size() == 1u;
                Slice timeSlice;
                for(auto& component : sliceComponents)
                {
                    if(n == 3)
                        break;
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

        /** create a RangeSlice out of an string
         *
         * Parse a comma separated list of slices and creates a vector of slices.
         * range slice syntax:
         *   - `begin:end:period`
         *   - a number ``N is equal to `N:N+1,1`
         *   - `:` is equal to `0:-1:1`
         *
         * If end is `-1` there is no defined end if the range.
         *
         * @param str Comma separated list of slices. Empty slices will be skipped.
         */
        inline std::vector<Slice> toRangeSlice(std::string const& str)
        {
            std::vector<Slice> result;
            auto const seqOfSlices = misc::splitString(str, ",");
            for(auto slice : seqOfSlices)
            {
                // skip empty slice strings
                if(slice.empty())
                    continue;
                else
                {
                    /* If the input is not empty extent the slice with one delimiter to get an empty string in cases
                     * where the slice ends with a delimiter without any other following characters.
                     */
                    slice += ":";
                }

                auto const sliceComponents = misc::splitString(slice, ":");
                PMACC_VERIFY_MSG(
                    !sliceComponents.empty(),
                    std::string("time slice without a defined element is not allowed") + str);

                // id of the component
                size_t n = 0;
                bool const sliceOnly = sliceComponents.size() == 1u;
                Slice rangeSlice;

                for(auto& component : sliceComponents)
                {
                    if(n == 3)
                        break;
                    // be sure that component it is a number or empty
                    PMACC_VERIFY_MSG(
                        component.empty() || detail::is_number(component),
                        std::string("value") + component + " in " + str + "is not a number");

                    rangeSlice.setValue(n, component);
                    if(sliceOnly)
                    {
                        // set end to begin + 1 to select only one slice
                        rangeSlice.values[1] = rangeSlice.values[0] + 1;
                    }
                    n++;
                }
                result.push_back(rangeSlice);
            }
            return result;
        }
    } // namespace pluginSystem
} // namespace pmacc
