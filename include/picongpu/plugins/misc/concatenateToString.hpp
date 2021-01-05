/* Copyright 2017-2021 Rene Widera
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

#include <string>
#include <numeric>


namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            /** concatenate all values of an string container
             *
             * @tparam T_Container type of the container
             *
             * @param vector source container (required interface: `begin(), end()`)
             * @param separator separator between two elements
             */
            template<typename T_Container>
            std::string concatenateToString(T_Container& container, std::string const& separator = ",")
            {
                return std::accumulate(
                    container.begin(),
                    container.end(),
                    std::string(),
                    [&](std::string& result, std::string& inString) {
                        return result.empty() ? inString : result + separator + inString;
                    });
            }
        } // namespace misc
    } // namespace plugins
} // namespace picongpu
