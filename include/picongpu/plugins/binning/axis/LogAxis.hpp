/* Copyright 2024 Tapish Narwal
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

#include "picongpu/plugins/binning/Axis.hpp"
#include "picongpu/plugins/binning/DomainInfo.hpp"
#include "picongpu/plugins/binning/UnitConversion.hpp"

#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace picongpu
{
    namespace plugins::binning
    {
        namespace axis
        {
            /**
             * Log axis with logarithmically sized bins.
             * Axis splitting is defined with min, max and n_bins.
             * A bin with edges a and b is a closed-open interval [a,b)
             * If overflow bins are enabled, allocates 2 extra bins, for under and overflow. These are bin index 0 and
             * (n_bins+2)-1
             */
            template<typename T_Attribute, typename T_AttrFunctor>
            class LogAxis
            {
            public:
                using T = T_Attribute;
                /**
                 * To avoid loss of precision, the type of the scaling depends on the Attribute type
                 * For integral types <4 bytes it is float, else it is double
                 * For floating point types it is the identity function
                 **/
                using ScalingType = std::
                    conditional_t<std::is_integral_v<T>, std::conditional_t<sizeof(T) == 4, float_X, double>, T>;

                AxisSplitting<T> axisSplit;
                /** Axis name, written out to OpenPMD */
                std::string label;
                /** Units(Dimensionality) of the axis */
                std::array<double, numUnits> units;
                std::vector<T> binWidths;

                /**
                 * @TODO store edges? Copmute once at the beginning and store for later to print at every
                 * iteration, also to be used in search based binning
                 */
                struct LogAxisKernel
                {
                    /** Function to place particle on axis, returns same type as min and max */
                    T_AttrFunctor getAttributeValue;
                    /**
                     * logMin and logMax values in the range of the binning. Values outside this range are
                     * placed in overflow bins
                     */
                    ScalingType logMin, logMax;
                    /** Number of bins in range */
                    uint32_t nBins;
                    ScalingType scaling;
                    /** Enable or disable allocation of extra bins for out of range particles*/
                    bool overflowEnabled;

                    constexpr LogAxisKernel(
                        T_AttrFunctor attrFunc,
                        AxisSplitting<T> axisSplit,
                        std::array<double, numUnits> unitsArr)
                        : getAttributeValue{attrFunc}
                        , overflowEnabled{axisSplit.enableOverflowBins}
                    {
                        // do conversion to PIC units here, if not auto
                        // auto picRange = toPICUnits(axSplit.range, unitsArr);
                        auto min = toPICUnits(axisSplit.m_range.min, unitsArr);
                        // toPICUnits might cause underflow
                        PMACC_VERIFY(0. < min);
                        logMin = std::log2(min);
                        logMax = std::log2(toPICUnits(axisSplit.m_range.max, unitsArr));

                        // do scaling calc here, on host and save it
                        scaling = static_cast<ScalingType>(axisSplit.nBins) / (logMax - logMin);

                        nBins = axisSplit.nBins;
                        if(axisSplit.enableOverflowBins)
                        {
                            nBins += 2;
                        }
                    }


                    template<typename T_Worker, typename T_Particle>
                    ALPAKA_FN_ACC uint32_t getBinIdx(
                        const DomainInfo& domainInfo,
                        const T_Worker& worker,
                        const T_Particle& particle,
                        bool& validIdx) const
                    {
                        auto val = getAttributeValue(domainInfo, worker, particle);

                        static_assert(
                            std::is_same<decltype(val), T>::value,
                            "The return type of the axisAttributeFunctor should be the same as the type of Axis "
                            "min/max ");

                        uint32_t binIdx = 0;
                        // @todo check if disableBinning is better
                        bool enableBinning = overflowEnabled;

                        if(static_cast<T>(0.) < val)
                        {
                            auto logVal = math::log2(val);

                            // @todo check for optimizations here
                            if(logVal >= logMin)
                            {
                                if(logVal < logMax)
                                {
                                    /**
                                     * Precision errors?
                                     * Is the math floor necessary?
                                     */
                                    binIdx = math::floor(((logVal - logMin) * scaling) + 1);
                                    if(!overflowEnabled)
                                    {
                                        enableBinning = true;
                                        binIdx = binIdx - 1;
                                    }
                                }
                                else
                                    binIdx = nBins - 1;
                            }
                        }

                        validIdx = validIdx && enableBinning;
                        return binIdx;
                    }
                };

                LogAxisKernel lAK;

                struct BinWidthKernel
                {
                    typename pmacc::HostDeviceBuffer<T, 1>::DBuffer::DataBoxType widthsDeviceBox;

                    BinWidthKernel(pmacc::HostDeviceBuffer<T, 1>& widths)
                        : widthsDeviceBox{widths.getDeviceBuffer().getDataBox()}
                    {
                    }

                    ALPAKA_FN_ACC T getBinWidth(uint32_t idx) const
                    {
                        return widthsDeviceBox(idx);
                    }
                };

                LogAxis(
                    AxisSplitting<T> axSplit,
                    T_AttrFunctor attrFunctor,
                    std::string label,
                    std::array<double, numUnits> unit_arr) // add type T to the default label string
                    : axisSplit{axSplit}
                    , label{label}
                    , units{unit_arr}
                    , lAK{attrFunctor, axisSplit, unit_arr}
                {
                    binWidths.reserve(axisSplit.nBins);
                    // Calculate the logarithmic spacing factor
                    ScalingType factor = std::pow(2., (lAK.logMax - lAK.logMin) / axisSplit.nBins);
                    // Calculate bin widths
                    ScalingType edge = axisSplit.m_range.min;
                    for(int i = 0; i < axisSplit.nBins; ++i)
                    {
                        ScalingType next_edge = factor * edge;
                        if constexpr(std::is_integral_v<T>)
                        {
                            // Ceiling because we have closed-open intervals
                            binWidths.emplace_back(math::ceil(next_edge - edge));
                        }
                        else
                        {
                            binWidths.emplace_back(next_edge - edge);
                        }
                        edge = next_edge;
                    }
                }


                constexpr uint32_t getNBins() const
                {
                    return lAK.nBins;
                }

                double getUnitConversion() const
                {
                    return getConversionFactor(units);
                }


                LogAxisKernel getAxisKernel() const
                {
                    return lAK;
                }

                BinWidthKernel getBinWidthKernel()
                {
                    // reset whenever binWidths changes
                    static bool set = false;
                    static pmacc::HostDeviceBuffer<T, 1> binWidthBuffer{axisSplit.nBins};
                    if(!set)
                    {
                        auto db = binWidthBuffer.getHostBuffer().getDataBox();
                        for(size_t i = 0; i < binWidths.size(); i++)
                        {
                            db[i] = binWidths[i];
                        }
                        binWidthBuffer.hostToDevice();
                        set = true;
                    }
                    return BinWidthKernel(binWidthBuffer);
                }

                /**
                 * @return bin edges in SI units
                 */
                std::vector<double> getBinEdgesSI()
                {
                    std::vector<double> binEdges;
                    binEdges.reserve(axisSplit.nBins + 1);
                    T edge = axisSplit.m_range.min;
                    for(size_t i = 0; i <= axisSplit.nBins; i++)
                    {
                        binEdges.emplace_back(toSIUnits(edge, units));
                        edge = edge + binWidths[i];
                    }
                    return binEdges;
                }
            };


            /**
             * @details Creates a log Axis
             * @tparam T_Attribute Type of the deposition functor (This is also the type of min, max and return type of
             * the attrFunctor and if these types dont match this will throw an error)
             * @param axisSplitting
             * @param functorDescription
             */
            template<typename T_Attribute, typename T_FunctorDescription>
            HINLINE auto createLog(AxisSplitting<T_Attribute> axSplit, T_FunctorDescription functorDesc)
            {
                static_assert(
                    std::is_same_v<typename T_FunctorDescription::QuantityType, T_Attribute>,
                    "Access functor return type and range type shuold be the same");
                /** this is doing an implicit conversion to T_attribute for min, max and scaling */
                return LogAxis<T_Attribute, typename T_FunctorDescription::FunctorType>(
                    axSplit,
                    functorDesc.functor,
                    functorDesc.name,
                    functorDesc.units);
            }

        } // namespace axis
    } // namespace plugins::binning
} // namespace picongpu
