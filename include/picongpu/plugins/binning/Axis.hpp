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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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
                bool enableOverflowBins;
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


            template<typename T_BinningFunctor>
            class GenericAxis
            {
            public:
                std::string label;
                std::array<double, numUnits> units;
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
                std::array<double, numUnits> units;
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

            template<typename T_FunctorDescription>
            HINLINE auto createBool(T_FunctorDescription functorDesc)
            {
                return BoolAxis<typename T_FunctorDescription::FunctorType>(functorDesc.functor, functorDesc.name);
            }

        } // namespace axis
    } // namespace plugins::binning
} // namespace picongpu
