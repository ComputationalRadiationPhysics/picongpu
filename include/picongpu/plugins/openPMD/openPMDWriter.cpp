/*
 * SPDX-FileCopyrightText: Franz Poeschel
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#if (ENABLE_OPENPMD == 1)

#    include <algorithm>
#    include <iterator>
#    include <sstream>

#    include <openPMD/openPMD.hpp>

namespace picongpu::openPMD
{
    std::string printAvailableExtensions()
    {
        std::vector variants_unfiltered = ::openPMD::getFileExtensions();
        std::vector<std::string> variants_filtered;
        std::copy_if(
            variants_unfiltered.begin(),
            variants_unfiltered.end(),
            std::back_inserter(variants_filtered),
            [](std::string const& s) { return s != "json" && s != "toml"; });
        if(variants_filtered.empty())
        {
            return "";
        }
        else
        {
            std::stringstream res;
            res << variants_filtered[0];
            for(size_t i = 1; i < variants_filtered.size(); ++i)
            {
                res << ", " << variants_filtered[i];
            }
            return res.str();
        }
    }
} // namespace picongpu::openPMD

#endif
