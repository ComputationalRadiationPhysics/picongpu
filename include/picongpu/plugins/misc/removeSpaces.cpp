/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "picongpu/plugins/misc/removeSpaces.hpp"

#include <algorithm>
#include <string>

namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            std::string removeSpaces(std::string value)
            {
                value.erase(std::remove(value.begin(), value.end(), ' '), value.end());

                return value;
            }
        } // namespace misc
    } // namespace plugins
} // namespace picongpu
