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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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

#include "picongpu/particles/atomicPhysics2/DebugHelper.hpp"
#include "picongpu/particles/atomicPhysics2/SuperCellField.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

#include <cstdint>
#include <iostream>
#include <string>

namespace picongpu::particles::atomicPhysics2::electronDistribution
{
    //! debug only, print content and bins of histogram to console, @attention serial and cpu build only!
    template<bool printOnlyOverSubscribed>
    struct PrintHistogramToConsole
    {
        template<typename T_Histogram>
        HINLINE void operator()(T_Histogram const& histogram, pmacc::DataSpace<picongpu::simDim> superCellIdx) const
        {
            constexpr uint32_t numBins = T_Histogram::numberBins;

            std::cout << "histogram " << superCellIdx.toString(",", "[]");
            std::cout << " base=" << histogram.getBase();
            std::cout << " numBins=" << T_Histogram::numberBins;
            std::cout << " maxE=" << T_Histogram::maxEnergy;
            std::cout << " overFlow: w0=" << histogram.getOverflowWeight() << std::endl;

            float_X centralEnergy;
            float_X binWidth;

            for(uint32_t i = 0u; i < numBins; i++)
            {
                if constexpr(printOnlyOverSubscribed)
                {
                    if(histogram.getBinWeight0(i) >= histogram.getBinDeltaWeight(i))
                        continue;
                }

                // binIndex
                std::cout << "\t " << i;

                // central bin energy [eV] and binWidth [eV]
                centralEnergy = histogram.getBinEnergy(i);
                binWidth = histogram.getBinWidth(i);

                std::cout << "(" << centralEnergy - binWidth / 2._X << ", " << centralEnergy + binWidth / 2._X
                          << "] :";

                // bin data, [w0, DeltaW, DeltaEnergy, binOverSubscribed]
                std::cout << " [w0, Dw, DE]: [";
                std::cout << histogram.getBinWeight0(i) << ", ";
                std::cout << histogram.getBinDeltaWeight(i) << ", ";
                std::cout << histogram.getBinDeltaEnergy(i) << "]";
                std::cout << std::endl;
            }
        }
    };

    /** holds a gridBuffer of the per-superCell localHistograms for atomicPhysics
     *
     * @attention histograms are uninitialized upon creation of the field, use .getDeviceBuffer().setValue() to init
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
