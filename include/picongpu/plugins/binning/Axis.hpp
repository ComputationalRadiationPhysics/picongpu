/* Copyright 2023 Tapish Narwal
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

#include "picongpu/plugins/binning/DomainInfo.hpp"
#include "picongpu/plugins/binning/UnitConversion.hpp"

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
             * @brief Holds the range in SI Units in axis space over which the binning will be done
             */
            template<typename T_Data>
            class Range
            {
            public:
                /** Minimum of binning range in SI Units */
                T_Data min;
                /** Maximum of binning range in SI Units */
                T_Data max;
                Range(T_Data minIn, T_Data maxIn) : min{minIn}, max{maxIn}
                {
                    PMACC_VERIFY(min < max);
                }
            };

            /**
             * @brief Holds the axis range in SI units and information on how this range is split into bins
             */
            template<typename T_Data>
            class AxisSplitting
            {
            public:
                /** Range object in SI units */
                Range<T_Data> m_range;
                /** Number of bins in range */
                uint32_t nBins;
                /** Enable or Disable overflow bins.
                 * Number of overflow bis is the responsibility of the axis implementaiton
                 * Defaults to true
                 */
                const bool enableOverflowBins;
                AxisSplitting(Range<T_Data> range, uint32_t numBins, bool enableOverflow = true)
                    : m_range{range}
                    , nBins{numBins}
                    , enableOverflowBins{enableOverflow}
                {
                }
            };

            // how to check if particle has this attribute? throw error at compile time
            // Overflow bins is the resposibilty of the axis implementation. Make nBins = user_defined_n_bins + 2
            // Bins which don't have an inherent size should return a bin width of 1 to not influence normalization.
            // @todo enforce some sort of interface on axis
            // @todo mark functions which are mandatory for each type of axis


            /**
             * Linear axis with contiguous fixed sized bins.
             * Axis splitting is defined with min, max and n_bins. Bin size = (max-min)/n_bins.
             * Bins are closed open [) intervals [min, min + size), [min + size, min + 2*size) ,..., [max-size, max).
             * Allocates 2 extra bins, for under and overflow. These are bin index 0 and (n_bins+2)-1
             */
            template<typename T_Attribute, typename T_AttrFunctor>
            class LinearAxis
            {
            public:
                using T = T_Attribute;
                AxisSplitting<T_Attribute> axisSplit;
                /** Axis name, written out to OpenPMD */
                std::string label;
                /** Units(Dimensionality) of the axis */
                std::array<double, 7> units;
                /**
                 * @TODO store edges? Copmute once at the beginning and store for later to print at every iteration,
                 * also to be used in search based binning
                 */
                std::vector<double> binEdges;
                struct LinearAxisKernel
                {
                    /** Function to place particle on axis, returns same type as min and max */
                    T_AttrFunctor getAttributeValue;
                    /**
                     * Min and max values in the range of the binning. Values outside this range are
                     * placed in overflow bins
                     */
                    T min, max;
                    /** Number of bins in range */
                    uint32_t nBins;
                    /** Using type depending on whether T is integer or floating point type to avoid precision loss */
                    using ScalingType = std::
                        conditional_t<std::is_integral_v<T>, std::conditional_t<sizeof(T) == 4, float_X, double>, T>;
                    ScalingType scaling;
                    /** Enable or disable allocation of extra bins for out of range particles*/
                    bool overflowEnabled;

                    constexpr LinearAxisKernel(T_AttrFunctor attrFunc, bool enableOverflowBins)
                        : getAttributeValue{attrFunc}
                        , overflowEnabled{enableOverflowBins}
                    {
                    }

                    /**
                     * @param n_bins User requested n_bins, actual nBins is n_bins + 2 for overflow and underflow bins
                     */
                    void initAxisSplit(T minR, T maxR, uint32_t n_bins, ScalingType scalingR)
                    {
                        min = minR;
                        max = maxR;
                        nBins = n_bins;
                        scaling = scalingR;
                    }

                    // can have a version without an idx also as the bin width is constant
                    constexpr T getBinWidth(uint32_t idx = 0) const
                    {
                        PMACC_ASSERT(idx < nBins);
                        return 1 / scaling;
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
                            std::is_same<decltype(val), decltype(min)>::value,
                            "The return type of the axisAttributeFunctor should be the same as the type of Axis "
                            "min/max ");
                        static_assert(
                            std::is_convertible<decltype(val), uint32_t>::value,
                            "Unable to cast to unit32_t for bin number");
                        uint32_t binIdx = 0;
                        bool enableBinning = overflowEnabled; // @todo check if disableBinning is better
                        // @todo check for optimizations here
                        if(val >= min)
                        {
                            if(val < max)
                            {
                                /**
                                 * Precision errors?
                                 * Is the math floor necessary?
                                 */
                                binIdx = math::floor(((val - min) * scaling) + 1);
                                if(!overflowEnabled)
                                {
                                    enableBinning = true;
                                    binIdx = binIdx - 1;
                                }
                            }
                            else
                                binIdx = nBins - 1;
                        }
                        validIdx = validIdx && enableBinning;
                        return binIdx;
                    }
                };

                LinearAxisKernel lAK;

                LinearAxis(
                    AxisSplitting<T_Attribute> axSplit,
                    T_AttrFunctor attrFunctor,
                    std::string label,
                    std::array<double, 7> units) // add type T to the default label string
                    : axisSplit{axSplit}
                    , label{label}
                    , units{units}
                    , lAK{attrFunctor, axisSplit.enableOverflowBins}
                {
                    initLAK();
                }


                /**
                 * @todo auto min max n_bins
                 */
                void initLAK()
                {
                    // do conversion to PIC units here, if not auto
                    // auto picRange = toPICUnits(axSplit.range, units);
                    auto min = toPICUnits(axisSplit.m_range.min, units);

                    auto max = toPICUnits(axisSplit.m_range.max, units);

                    // do scaling calc here, on host and save it
                    auto scaling = static_cast<decltype(max)>(axisSplit.nBins) / (max - min);


                    auto nBins = axisSplit.nBins;
                    if(axisSplit.enableOverflowBins)
                    {
                        nBins += 2;
                    }

                    lAK.initAxisSplit(min, max, nBins, scaling);
                }

                constexpr uint32_t getNBins() const
                {
                    return lAK.nBins;
                }

                double getUnitConversion() const
                {
                    return get_conversion_factor(units);
                }


                LinearAxisKernel getAxisKernel() const
                {
                    return lAK;
                }

                /**
                 * @return bin edges in SI units
                 */
                std::vector<double> getBinEdgesSI()
                {
                    auto binWidth = 1. / lAK.scaling;
                    // user_nBins+1 edges
                    for(size_t i = 0; i <= lAK.nBins - 2; i++)
                    {
                        binEdges.emplace_back(toSIUnits(lAK.min + i * binWidth, units));
                    }
                    return binEdges;
                }
            };

            template<typename T_BinningFunctor>
            class GenericAxis
            {
            public:
                std::string label;
                std::array<double, 7> units;
                struct GenericAxisKernel
                {
                    uint32_t n_bins;
                    T_BinningFunctor getBinIdx;

                    constexpr GenericAxisKernel(uint32_t n_bins, T_BinningFunctor binFunctor)
                        : n_bins{n_bins}
                        , getBinIdx{binFunctor}
                    {
                    }
                };
                GenericAxisKernel gAK;

                GenericAxis(uint32_t n_bins, T_BinningFunctor binFunctor) : gAK{n_bins, binFunctor}
                {
                }

                constexpr uint32_t getNBins() const
                {
                    return gAK.n_bins;
                }
            };

            template<typename T_AttrFunctor>
            class BoolAxis
            {
            public:
                std::string label;
                std::array<double, 7> units;
                struct BoolAxisKernel
                {
                    uint32_t n_bins;
                    T_AttrFunctor getAttributeValue;

                    constexpr BoolAxisKernel(T_AttrFunctor attrFunctor) : n_bins{2u}, getAttributeValue{attrFunctor}
                    {
                    }

                    template<typename T_Worker, typename T_Particle>
                    ALPAKA_FN_ACC uint32_t
                    getBinIdx(const DomainInfo& domainInfo, const T_Worker& worker, const T_Particle& particle) const
                    {
                        // static cast to bool ?
                        // bool val = getAttributeValue(worker, particle);
                        return 0;
                    }
                };

                BoolAxisKernel bAK;

                BoolAxis(T_AttrFunctor attrFunctor, std::string label = "Bool_Axis") : bAK{attrFunctor}, label{label}
                {
                }

                constexpr uint32_t getNBins() const
                {
                    return bAK.n_bins;
                }

                BoolAxisKernel getAxisKernel() const
                {
                    return bAK;
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
            HINLINE auto createLinear(AxisSplitting<T_Attribute> axSplit, T_FunctorDescription functorDesc)
            {
                static_assert(
                    std::is_same_v<typename T_FunctorDescription::QuantityType, T_Attribute>,
                    "Access functor return type and range type shuold be the same");
                /** this is doing an implicit conversion to T_attribute for min, max and scaling */
                return LinearAxis<T_Attribute, typename T_FunctorDescription::FunctorType>(
                    axSplit,
                    functorDesc.functor,
                    functorDesc.name,
                    functorDesc.units);
            }

            template<typename T_FunctorDescription>
            HINLINE auto createBool(T_FunctorDescription functorDesc)
            {
                return BoolAxis<typename T_FunctorDescription::FunctorType>(functorDesc.functor, functorDesc.name);
            }

        } // namespace axis
    } // namespace plugins::binning
} // namespace picongpu
