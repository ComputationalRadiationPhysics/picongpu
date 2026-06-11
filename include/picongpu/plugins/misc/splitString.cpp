/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "picongpu/plugins/misc/splitString.hpp"

#include <regex>
#include <string>
#include <vector>

namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            std::vector<std::string> splitString(std::string const& input, std::string const& regex)
            {
                std::regex re(regex);
                // passing -1 as the submatch index parameter performs splitting
                std::sregex_token_iterator first{input.begin(), input.end(), re, -1};
                std::sregex_token_iterator last;

                return {first, last};
            }
        } // namespace misc
    } // namespace plugins
} // namespace picongpu
