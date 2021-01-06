/* Copyright 2016-2021 Rene Widera
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

#include <iostream>
#include <string>
#include <stdexcept>
#include <sstream>

namespace pmacc
{
    namespace
    {
        /** abort program with an exception
         *
         * This function always throws a `runtime_error`.
         *
         * @param exp evaluated expression
         * @param filename name of the broken file
         * @param lineNumber line in file
         * @param msg user defined error message
         */
        void abortWithError(
            const std::string exp,
            const std::string filename,
            const uint32_t lineNumber,
            const std::string msg = std::string())
        {
            std::stringstream line;
            line << lineNumber;

            throw std::runtime_error(
                "expression (" + exp + ") failed in file (" + filename + ":" + line.str() + ") : " + msg);
        }
    } // namespace
} // namespace pmacc
