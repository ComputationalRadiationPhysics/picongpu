/* Copyright 2014-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Franz Poeschel
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
#include <openPMD/openPMD.hpp>

#include <vector>
#include <map>

namespace picongpu
{
    namespace openPMD
    {
        /** convert PIConGPU dimension unit into a corresponding openPMD map
         *
         * @param unitDimension PIConGPU dimension vector
         * @return openPMD-api dimension map
         */
        inline auto convertToUnitDimension(std::vector<float_64> const& unitDimension)
        {
            PMACC_ASSERT(unitDimension.size() == 7); // seven openPMD base units
            constexpr ::openPMD::UnitDimension openPMDUnitDimensions[7]
                = {::openPMD::UnitDimension::L,
                   ::openPMD::UnitDimension::M,
                   ::openPMD::UnitDimension::T,
                   ::openPMD::UnitDimension::I,
                   ::openPMD::UnitDimension::theta,
                   ::openPMD::UnitDimension::N,
                   ::openPMD::UnitDimension::J};
            std::map<::openPMD::UnitDimension, double> unitMap;
            for(unsigned i = 0; i < 7; ++i)
            {
                unitMap[openPMDUnitDimensions[i]] = unitDimension[i];
            }

            return unitMap;
        }
    } // namespace openPMD
} // namespace picongpu
