/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file implements a local histogram field for each superCell
 *
 */

#pragma once

#include "picongpu/particles/atomicPhysics/SuperCellField.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

#include <cstdint>
#include <iostream>
#include <string>

namespace picongpu::particles::atomicPhysics::electronDistribution
{
    namespace enums
    {
        enum struct BinSelection : uint8_t
        {
            OnlyOverSubscribed,
            All
        };
    } // namespace enums

    /** debug only, print content and bins of histogram to console
     *
     * @attention only creates output if atomicPhysics debug setting CPU_OUTPUT_ACTIVE == True
     * @attention only useful if compiling for serial backend, otherwise output for different histograms will
     * interleave
     */
    template<enums::BinSelection T_BinSelection>
    struct PrintHistogramToConsole
    {
        //! cpu version
        template<typename T_Acc, typename T_Histogram>
        HDINLINE auto operator()(
            T_Acc const&,
            T_Histogram const& histogram,
            pmacc::DataSpace<picongpu::simDim> superCellIdx) const
            -> std::enable_if_t<std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
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
                if constexpr(T_BinSelection == enums::BinSelection::OnlyOverSubscribed)
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

        //! gpu verison, does nothing
        template<typename T_Acc, typename T_Histogram>
        HDINLINE auto operator()(
            T_Acc const&,
            T_Histogram const& histogram,
            pmacc::DataSpace<picongpu::simDim> superCellIdx) const
            -> std::enable_if_t<!std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
        }
    };

    /** holds a gridBuffer of the per-superCell histograms for atomicPhysics
     *
     * @attention histograms are uninitialized upon creation of the field, use .getDeviceBuffer().setValue() to init
     */
    template<typename T_Histogram, typename T_MappingDescription>
    struct LocalHistogramField : SuperCellField<T_Histogram, T_MappingDescription, false /*no guards*/>
    {
        using Histogram = T_Histogram;

    private:
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
            return histogramType + "_HistogramField";
        }
    };

} // namespace picongpu::particles::atomicPhysics::electronDistribution
