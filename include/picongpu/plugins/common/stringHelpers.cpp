/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "picongpu/plugins/common/stringHelpers.hpp"

namespace picongpu
{
    namespace helper
    {
        /** Return the current date as string
         *
         * @param format, @see http://www.cplusplus.com/reference/ctime/strftime/
         * @return std::string with formatted date
         */
        std::string getDateString(std::string format)
        {
            time_t rawtime;
            struct tm* timeinfo;
            size_t const maxLen = 30;
            char buffer[maxLen];

            time(&rawtime);
            timeinfo = localtime(&rawtime);

            strftime(buffer, maxLen, format.c_str(), timeinfo);

            std::stringstream dateString;
            dateString << buffer;

            return dateString.str();
        }
    } // namespace helper
} // namespace picongpu
