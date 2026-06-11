/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/verify.hpp"

#include <array>
#include <charconv>
#include <cstdint>
#include <string>

namespace pmacc
{
    namespace pluginSystem
    {
        struct Slice
        {
            /** time slice configuration
             *
             * 0 = begin of the interval
             * 1 = end of the interval
             * 2 = period
             */
            std::array<uint32_t, 3> values;

            std::string toString() const
            {
                std::string result;
                result = std::to_string(values[0]) + ":" + std::to_string(values[1]) + ":" + std::to_string(values[2]);
                return result;
            }

            /** set the value
             *
             * if str is empty the default value for the given index is selected
             *
             * @param idx index to set, range [0,3)
             * @param str value to set, can be empty
             */
            void setValue(uint32_t const idx, std::string const& str)
            {
                if(!str.empty())
                {
                    auto value = static_cast<uint32_t>(std::stoul(str));
                    PMACC_VERIFY_MSG(!(idx == 2 && value == 0), "Zero is not a valid period");
                    values.at(idx) = value;
                }
            }

            /** Constructor
             *
             * Select any time step in the range [start,end].
             *
             * @param start first time step taken into account
             * @param end last time step taken into account
             * @param period select any period(st) time step between start and end
             */
            Slice(uint32_t start = 0, uint32_t end = uint32_t(-1), uint32_t period = 1)
                : /* default: start:end:period
                   * -1 stored as unsigned is the highest available unsigned integer
                   */
                values({start, end, period})
            {
            }
        };
    } // namespace pluginSystem
} // namespace pmacc
