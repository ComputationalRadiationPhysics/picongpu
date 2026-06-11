/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            /** Get text names of vector components
             *
             * For 1-4 and 6 components use predefined names,
             * for other amounts use generic different names
             *
             * @param numComponents number of components
             */
            std::vector<std::string> getComponentNames(uint32_t numComponents);

        } // namespace misc
    } // namespace plugins
} // namespace picongpu
