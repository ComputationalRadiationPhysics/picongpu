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
#include <vector>


namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            /** split a string in a vector of strings
             *
             * Based on Stack Overflow post:
             *   source: https://stackoverflow.com/a/28142357
             *   author: Marcin
             *   date: Jan 25 '15
             *
             * @param input string to split
             * @param regex separator between two elements
             */
            std::vector<std::string> splitString(std::string const& input, std::string const& regex = ",");
        } // namespace misc
    } // namespace plugins
} // namespace picongpu
