/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <string>

namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            /** removes all spaces within a string
             *
             * @param value input string
             * @return string without any spaces
             */
            std::string removeSpaces(std::string value);
        } // namespace misc
    } // namespace plugins
} // namespace picongpu
