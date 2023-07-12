/* Copyright 2022-2023 Brian Marre
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

/** @file implements the local electron histogram field for each superCell
 *
 */

#pragma once

#include "picongpu/particles/atomicPhysics2/SuperCellField.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics2::electronDistribution
{
    /**@class holds a gridBuffer of the per-superCell localHistograms for atomicPhysics
     *
     * @attention histograms are uninitialized upon creation of the field,
     *  use .getDeviceBuffer().setValue() to init
     */
    template<typename T_Histogram, typename T_MappingDescription>
    struct LocalHistogramField : SuperCellField<T_Histogram, T_MappingDescription, false /*no guards*/>
    {
        using Histogram = T_Histogram;

    private:
        /// @todo should these be private?
        //! type of physical particle represented in histogram, usually "Electron" or "Photon"
        std::string histogramType;

    public:
        LocalHistogramField(T_MappingDescription const& mappingDesc, std::string const histogramType)
            : SuperCellField<T_Histogram, T_MappingDescription, false /*no guards*/>(mappingDesc)
            , histogramType(histogramType)
        {
        }

        //! required by ISimulationData
        std::string getUniqueId() override
        {
            return histogramType + "_localHistogramField";
        }
    };

} // namespace picongpu::particles::atomicPhysics2::electronDistribution
