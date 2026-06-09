/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2023-2024 Tapish Narwal
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
             * Linear axis with contiguous fixed sized bins.
             * Axis splitting is defined with min, max and n_bins. Bin size = (max-min)/n_bins.
             * Bins are closed open [) intervals [min, min + size), [min + size, min + 2*size) ,..., [max-size, max).
             * Allocates 2 extra bins, for under and overflow. These are bin index 0 and (n_bins+2)-1
             * T_Attribute is a Numeric/Arithmetic type
             */
            template<typename T_Attribute, typename T_AttrFunctor>
            class LinearAxis
            {
            private:
                // Depends on other class members being initialized
                void initBinEdges()
                {
                    // user_nBins+1 edges
                    binEdges.reserve(axisSplit.nBins + 1);
                    auto const binWidth
                        = static_cast<double>(axisSplit.m_range.getRange() / static_cast<double>(axisSplit.nBins));
                    for(size_t i = 0; i <= axisSplit.nBins; i++)
                    {
                        binEdges.emplace_back(axisSplit.m_range.min + i * binWidth);
                    }
                }

            public:
                struct LinearAxisKernel
                {
                    /** Function to place particle on axis, returns same type as min and max */
                    T_AttrFunctor getAttributeValue;
                    /**
                     * Min and max values in the range of the binning. Values outside this range are
                     * placed in overflow bins
                     * Range in PIC units
                     */
                    Range<T_Attribute> picRange;
                    /** Number of bins in range, including overflow bins */
                    uint32_t nBins;
                    /** For integral types = axSplit.nBins (EXCLUDING overflow bins)
                     * for floating point types = axSplit.nBins/picRange.getRange()
                     */
                    T_Attribute scaling;
                    /** Enable or disable allocation of extra bins for out of range particles*/
                    bool overflowEnabled;

                    constexpr LinearAxisKernel(
                        T_AttrFunctor const& attrFunc,
                        AxisSplitting<T_Attribute> const& axisSplit,
                        std::array<double, numUnits> const& unitsArr)
                        : getAttributeValue{attrFunc}
                        , picRange{toPICUnits(axisSplit.m_range.min, unitsArr), toPICUnits(axisSplit.m_range.max, unitsArr)}
                        , nBins{axisSplit.enableOverflowBins ? axisSplit.nBins + 2 : axisSplit.nBins}
                        , scaling{static_cast<T_Attribute>(axisSplit.nBins)}
                        , overflowEnabled{axisSplit.enableOverflowBins}
                    {
                        if constexpr(std::is_floating_point_v<T_Attribute>)
                        {
                            scaling /= picRange.getRange();
                        }
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
                                if constexpr(std::is_integral_v<T_Attribute>)
                                {
                                    // using only integer operations
                                    using PromotedType
                                        = std::conditional_t<std::is_signed_v<T_Attribute>, int64_t, uint64_t>;

                                    PromotedType const diff = val - picRange.min;
                                    PromotedType const scaled_diff = diff * scaling;
                                    binIdx = scaled_diff / picRange.getRange();
                                }
                                else
                                {
                                    // cast to bin index works like a floor
                                    binIdx = (val - picRange.min) * scaling;
                                }
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

                using Type = T_Attribute;

                AxisSplitting<T_Attribute> axisSplit;
                /** Axis name, written out to OpenPMD */
                std::string label;
                /** Units(Dimensionality) of the axis */
                std::array<double, numUnits> units;
                LinearAxisKernel lAK;
                std::vector<double> binEdges;

                LinearAxis(
                    AxisSplitting<T_Attribute> const& axSplit,
                    T_AttrFunctor const& attrFunctor,
                    std::string const& label,
                    std::array<double, numUnits> const& units) // add type T_Attribute to the default label string
                    : axisSplit{axSplit}
                    , label{label}
                    , units{units}
                    , lAK{attrFunctor, axSplit, units}
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

                LinearAxisKernel getAxisKernel() const
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
             * @details Creates a linear axis Bin width is (max-min)/n_bins
             * @tparam T_Attribute Type of the deposition functor (This is also the type of min, max and return type of
             * the attrFunctor and if these types dont match this will throw an error)
             * @param axisSplitting
             * @param functorDescription
             */
            template<typename T_Attribute, typename T_FunctorDescription>
            HINLINE auto createLinear(
                AxisSplitting<T_Attribute> const& axSplit,
                T_FunctorDescription const& functorDesc)
            {
                static_assert(
                    std::is_same_v<typename T_FunctorDescription::QuantityType, T_Attribute>,
                    "Access functor return type and range type should be the same");
                /** this is doing an implicit conversion to T_attribute for min, max and scaling */
                return LinearAxis<T_Attribute, typename T_FunctorDescription::FunctorType>(
                    axSplit,
                    functorDesc.functor,
                    functorDesc.name,
                    functorDesc.units);
            }

        } // namespace axis
    } // namespace plugins::binning
} // namespace picongpu
