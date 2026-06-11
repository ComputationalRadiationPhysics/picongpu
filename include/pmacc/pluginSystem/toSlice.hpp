/*
 * SPDX-FileCopyrightText: Rene Widera, Julian Lenz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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

        /** create a Slice out of an string
         *
         * Parse a comma separated list of slices and creates a vector of slices.
         * range slice syntax:
         *   - `begin:end:period`
         *   - `:` is equal to `0:-1:1`
         *
         * If end is `-1` there is no defined end if the range.
         * As a shorthand syntax, we allow a single int to be given. In such cases, `fillDefaults` is called to arrange
         * the values correctly.
         *
         * @param str Comma separated list of slices. Empty slices will be skipped.
         */
        inline std::vector<Slice> toSlice(std::string const& str, auto const fillDefaults)
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
                uint32_t n = 0;
                bool const sliceOnly = sliceComponents.size() == 1u;
                Slice rangeSlice;

                for(auto& component : sliceComponents)
                {
                    if(n == 3)
                        break;
                    // be sure that component it is a number or empty
                    PMACC_VERIFY_MSG(
                        component.empty() || detail::is_number(component) || (n == 1 && component == "-1"),
                        std::string("value") + component + " in " + str + "is not a number");

                    rangeSlice.setValue(n, component);
                    if(sliceOnly)
                    {
                        rangeSlice = fillDefaults(rangeSlice);
                    }
                    n++;
                }
                result.push_back(rangeSlice);
            }
            return result;
        }

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
            return toSlice(
                str,
                [](Slice const& slice)
                {
                    Slice s = slice;
                    // If a single number was given, the first value has been filled.
                    // In our case we want to interpret this as a period, so we put it to the last slot.
                    s.values[2] = s.values[0];
                    s.values[0] = 0;
                    s.values[1] = static_cast<uint32_t>(-1);
                    return s;
                });
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
            return toSlice(
                str,
                [](Slice const& slice)
                {
                    Slice s = slice;
                    // If a single number was given, the first value has been filled.
                    // In our case we want to interpret this as a single slice, so it's N:N+1:1.
                    s.values[1] = s.values[0] + 1;
                    s.values[2] = 1;
                    return s;
                });
        }

    } // namespace pluginSystem
} // namespace pmacc
