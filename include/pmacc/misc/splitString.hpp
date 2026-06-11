/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <string>
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
        std::vector<std::string> splitString(std::string const& input, std::string const& delimiter = ",");
    } // namespace misc
} // namespace pmacc
