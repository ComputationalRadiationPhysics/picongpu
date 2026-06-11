/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera, Franz Poeschel
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#if (ENABLE_OPENPMD == 1)

#    include "picongpu/defines.hpp"

#    include <map>
#    include <vector>

#    include <openPMD/openPMD.hpp>

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

#endif
