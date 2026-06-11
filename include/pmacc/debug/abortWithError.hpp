/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

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
        inline void abortWithError(
            std::string const exp,
            std::string const filename,
            uint32_t const lineNumber,
            std::string const msg = std::string())
        {
            std::stringstream line;
            line << lineNumber;

            throw std::runtime_error(
                "expression (" + exp + ") failed in file (" + filename + ":" + line.str() + ") : " + msg);
        }
    } // namespace
} // namespace pmacc
