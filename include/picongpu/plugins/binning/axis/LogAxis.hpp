/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2024-2024 Tapish Narwal
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

#pragma once

#include "picongpu/plugins/binning/UnitConversion.hpp"
#include "picongpu/plugins/binning/axis/Axis.hpp"

#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
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
             * Similar to geompace from numpy
             * Axis splitting is defined with min, max and n_bins.
             * A bin with edges a and b is a closed-open interval [a,b)
             * If overflow bins are enabled, allocates 2 extra bins, for under and overflow. These are bin index 0 and
             * (n_bins+2)-1
             * WARNING: For integral valued axes, the LogAxis always suffers from floating point erros, and can for
             * certain axis splittings and value combinations under or over estimate the bin index
             */
            template<typename T_Attribute, typename T_AttrFunctor>
            class LogAxis
            {
                // Depends on other class members being initialized
                void initBinEdges()
                {
                    binEdges.reserve(axisSplit.nBins + 1);
                    double edge = axisSplit.m_range.min;
                    for(size_t i = 0; i <= axisSplit.nBins; i++)
                    {
                        binEdges.emplace_back(edge);
                        edge = edge * geomFactor;
                    }
                }

            public:
                using Type = T_Attribute;
                /**
                 * This type is used for the scaling and to hold the log of the attribute values
                 * Scaling is the multiplication factor used to scale (val-min) to find the bin idx
                 * The type of the scaling depends on the Attribute type and is set to provide "reasonable" precision
                 * For integral types <= 4 bytes it is float, else it is double
                 * For floating point types it is the identity function
                 **/
                using ScalingType = std::conditional_t<
                    std::is_integral_v<T_Attribute>,
                    std::conditional_t<sizeof(T_Attribute) <= 4, float_X, double>,
                    T_Attribute>;

                AxisSplitting<T_Attribute> axisSplit;
                /** Axis name, written out to OpenPMD */
                std::string label;
                /** Units(Dimensionality) of the axis */
                std::array<double, numUnits> units;
                // Geometric factor (unitless)
                double geomFactor;
                /** axisSplit.nBins + 1 bin edges in SI units */
                std::vector<double> binEdges;

                /**
                 * @TODO store edges? Copmute once at the beginning and store for later to print at every
                 * iteration, also to be used in search based binning
                 */
                struct LogAxisKernel
                {
                    /** Function to place particle on axis, returns same type as min and max */
                    T_AttrFunctor getAttributeValue;
                    /**
                     * Min and max values in the range of the binning. Values outside this range are
                     * placed in overflow bins
                     * Range in PIC units
                     * Usage in log axis assumes toPICUnits doesnt flip sign
                     */
                    Range<T_Attribute> picRange;
                    /** Enable or disable allocation of extra bins for out of range particles*/
                    bool overflowEnabled;
                    /** Number of bins in range */
                    uint32_t nBins;
                    ScalingType scaling;

                    constexpr LogAxisKernel(
                        T_AttrFunctor const& attrFunc,
                        AxisSplitting<T_Attribute> const& axisSplit,
                        std::array<double, numUnits> const& unitsArr)
                        : getAttributeValue{attrFunc}
                        , picRange{toPICUnits(axisSplit.m_range.min, unitsArr), toPICUnits(axisSplit.m_range.max, unitsArr)}
                        , overflowEnabled{axisSplit.enableOverflowBins}
                        , nBins{overflowEnabled ? axisSplit.nBins + 2 : axisSplit.nBins}
                        , scaling{
                              static_cast<ScalingType>(axisSplit.nBins)
                              / static_cast<ScalingType>(
                                  std::log2(static_cast<ScalingType>(picRange.max) / picRange.min))}
                    {
                        // toPICUnits might cause underflow
                        PMACC_VERIFY(picRange.min != 0);
                    }

                    template<typename... Args>
                    ALPAKA_FN_HOST_ACC std::pair<bool, uint32_t> getBinIdx(Args const&... args) const
                    {
                        auto val = getAttributeValue(args...);

                        static_assert(
                            std::is_same<decltype(val), T_Attribute>::value,
                            "The return type of the axisAttributeFunctor should be the same as the type of Axis "
                            "min/max ");

                        uint32_t binIdx = 0;
                        // @todo check if disableBinning is better
                        bool enableBinning = overflowEnabled;

                        // @todo check for optimizations here
                        if(val >= picRange.min)
                        {
                            if(val < picRange.max)
                            {
                                /** we know min and max have the same sign, and since val is between them it also has
                                 * the same sign
                                 */
                                auto const x = math::log2(val / picRange.min);

                                // Cast to bin index works like a floor
                                binIdx = x * scaling;

                                if(overflowEnabled)
                                {
                                    // shift for 0th overflow bin
                                    ++binIdx;
                                    // binning is always done for all vals if overflow bins are enabled
                                }
                                else
                                {
                                    // no shift required
                                    // val is in range, so enable binning
                                    enableBinning = true;
                                }
                            }
                            else
                                binIdx = nBins - 1;
                        }
                        return {enableBinning, binIdx};
                    }
                };

                LogAxisKernel lAK;

                LogAxis(
                    AxisSplitting<T_Attribute> axSplit,
                    T_AttrFunctor attrFunctor,
                    std::string label,
                    std::array<double, numUnits> unit_arr) // add type T_Attribute to the default label string
                    : axisSplit{axSplit}
                    , label{label}
                    , units{unit_arr}
                    , lAK{attrFunctor, axisSplit, unit_arr}
                    , geomFactor{std::pow(
                          static_cast<double>(axisSplit.m_range.max) / axisSplit.m_range.min,
                          1.0 / static_cast<double>(axSplit.nBins))}
                {
                    initBinEdges();
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

                /**
                 * @return bin edges in SI units
                 */
                std::vector<double> getBinEdgesSI() const
                {
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
            HINLINE auto createLog(AxisSplitting<T_Attribute> const& axSplit, T_FunctorDescription const& functorDesc)
            {
                if(axSplit.m_range.min < 0 != axSplit.m_range.max < 0)
                {
                    throw std::domain_error("Range can't include zero");
                }

                static_assert(
                    std::is_same_v<typename T_FunctorDescription::QuantityType, T_Attribute>,
                    "Access functor return type and range type should be the same");
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
