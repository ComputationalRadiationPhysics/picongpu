/**
 * Copyright 2015-2016 Axel Huebl
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
#include <sstream>
#include <ctime>
#include <vector>
#include <list>
#include <algorithm>
#include <iostream>

namespace picongpu
{
namespace helper
{
    /** Return the current date as string
     *
     * \param format, \see http://www.cplusplus.com/reference/ctime/strftime/
     * \return std::string with formatted date
     */
    std::string getDateString( std::string format );

    /** Create array of c-strings suitable for libSplash
     *
     * Convert a std::list of strings to a format that is suitable to
     * be written into libSplash (concated and padded array of constant
     * c-strings). Strings will be padded to longest string.
     *
     * Independent of the padding you chose, the strings will be '\0'
     * separated & terminated. \0 padding is default and recommended.
     */
    class GetSplashArrayOfString
    {
    private:
        // compare two std::string by their size
        struct CompStrBySize
        {
            bool operator()( std::string i, std::string j )
            {
                return i.size() < j.size();
            }
        };

    public:
        // resulting type containing all attributes for a libSplash write call
        struct Result
        {
            size_t maxLen;                // size of the longest string
            std::vector<char> buffers;    // all of same length lenMax

            Result() : maxLen(0)
            {}
        };

        Result operator()(
            std::list<std::string> listOfStrings,
            char padding = '\0'
        );
    };

} // namespace helper
} // namespace picongpu
