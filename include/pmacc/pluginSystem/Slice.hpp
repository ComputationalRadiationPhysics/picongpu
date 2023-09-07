/* Copyright 2018-2023 Rene Widera
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

#include "pmacc/verify.hpp"

#include <array>
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
                    uint32_t value = std::stoul(str);
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
