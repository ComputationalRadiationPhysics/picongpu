/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <algorithm>
#include <ctime>
#include <iostream>
#include <list>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace picongpu
{
    namespace helper
    {
        /** Return the current date as string
         *
         * @param format, @see http://www.cplusplus.com/reference/ctime/strftime/
         * @return std::string with formatted date
         */
        std::string getDateString(std::string format);
    } // namespace helper
} // namespace picongpu
