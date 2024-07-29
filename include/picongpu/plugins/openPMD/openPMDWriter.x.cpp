/* Copyright 2022-2023 Franz Poeschel, Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#if(ENABLE_OPENPMD == 1)

#    include "picongpu/plugins/openPMD/openPMDWriter.hpp"

#    include "picongpu/plugins/PluginRegistry.hpp"
#    include "picongpu/plugins/multi/multi.hpp"

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


PIC_REGISTER_PLUGIN(picongpu::plugins::multi::Master<picongpu::openPMD::openPMDWriter>);

#endif
