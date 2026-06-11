/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/debug/param.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

#include <cstdint>
#include <iomanip>
#include <iostream>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /** debug only, write content of rate cache to console
     *
     * @attention only useful of compiling for serial backend, otherwise different RejectionProbabilityCache's outputs
     *  will interleave
     */
    template<bool printOnlyOversubscribed>
    struct PrintRejectionProbabilityCacheBinToConsole
    {
        template<typename T_Acc, typename T_RejectionProbabilityCache>
        HDINLINE auto operator()(
            T_Acc const&,
            T_RejectionProbabilityCache const& rejectionProbabilityCache,
            pmacc::DataSpace<picongpu::simDim> superCellIdx) const
            -> std::enable_if_t<std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
            constexpr uint32_t numBins = T_RejectionProbabilityCache::numberBins;

            // check if overSubscribed
            bool overSubscription = false;
            for(uint32_t i = 0u; i < numBins; i++)
            {
                if(rejectionProbabilityCache.getRejectionProbabilityBin(i) > 0._X)
                    overSubscription = true;
            }

            // print content
            std::cout << "rejectionProbabilityCache_Bin " << superCellIdx.toString(",", "[]");
            std::cout << " oversubcribed: " << ((overSubscription) ? "true" : "false") << std::endl;
            std::cout << "Bins:" << std::endl;
            for(uint32_t i = 0u; i < numBins; i++)
            {
                if constexpr(printOnlyOversubscribed)
                {
                    if(rejectionProbabilityCache.getRejectionProbabilityBin(i) < 0._X)
                        continue;
                }

                std::cout << "\t\t" << i << ":[ " << std::setw(10) << std::scientific
                          << rejectionProbabilityCache.getRejectionProbabilityBin(i) << std::defaultfloat << " ]"
                          << std::endl;
            }
        }

        template<typename T_Acc, typename T_RejectionProbabilityCache>
        HDINLINE auto operator()(
            T_Acc const&,
            T_RejectionProbabilityCache const& rejectionProbabilityCache,
            pmacc::DataSpace<picongpu::simDim> superCellIdx) const
            -> std::enable_if_t<!std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
        }
    };

    /** cache of rejection probability p_bin for over subscribed bins,
     *
     * p_Bin = (binDeltaWeight - binWeight0)/binDeltaWeight
     *
     * @tparam T_numberBins number of bin entries in cache
     *
     * @attention invalidated every time the electron spectrum
     */
    template<uint32_t T_numberBins>
    class RejectionProbabilityCache_Bin
    {
    public:
        static constexpr uint32_t numberBins = T_numberBins;

    private:
        float_X rejectionProbabilityBin[numberBins] = {-1._X}; // unitless

        //! @attention only active by debug setting
        HDINLINE static bool outOfRangeBinIndex(uint32_t const binIndex)
        {
            if constexpr(picongpu::atomicPhysics::debug::rejectionProbabilityCache::BIN_INDEX_RANGE_CHECK)
                if(binIndex >= numberBins)
                {
                    printf("atomicPhysics ERROR: out of range bin index in call to RejectionProbabilityCache_Bin\n");
                    return true;
                }
            return false;
        }

    public:
        /** set cache entry
         *
         * @param binIndex
         * @param rejectionProbability rejectionProbability of bin
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         * @attention only use if only ever one thread accesses each rate cache entry!
         */
        HDINLINE void setBin(uint32_t const binIndex, float_X const rejectionProbability)
        {
            if(outOfRangeBinIndex(binIndex))
                return;

            rejectionProbabilityBin[binIndex] = rejectionProbability;
        }

        /** get cached rejectionProbability for a bin
         *
         * @param binIndex
         * @return rejectionProbability rejectionProbability of bin
         *
         * @attention no range checks outside a debug compile, invalid memory access on failure
         */
        HDINLINE float_X getRejectionProbabilityBin(uint32_t const binIndex) const
        {
            if(outOfRangeBinIndex(binIndex))
                return -1._X;

            return rejectionProbabilityBin[binIndex];
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
