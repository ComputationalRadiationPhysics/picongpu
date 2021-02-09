/* Copyright 2017-2021 Rene Widera
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

#include <string>
#include <regex>
#include <vector>


namespace pmacc
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
        HINLINE std::vector<std::string> splitString(std::string const& input, std::string const& delimiter = ",")
        {
            std::regex re(delimiter);
            // passing -1 as the submatch index parameter performs splitting
            std::sregex_token_iterator first{input.begin(), input.end(), re, -1};
            std::sregex_token_iterator last;

            return {first, last};
        }
    } // namespace misc
} // namespace pmacc
