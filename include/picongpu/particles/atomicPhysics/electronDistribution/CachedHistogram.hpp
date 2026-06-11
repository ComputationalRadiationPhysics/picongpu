/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/memory/Array.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::kernel
{
    template<uint32_t T_size>
    struct CachedHistogram
    {
        pmacc::memory::Array<float_X, T_size> energy;
        pmacc::memory::Array<float_X, T_size> binWidth;
        pmacc::memory::Array<float_X, T_size> density;

        static constexpr uint32_t size = T_size;

        constexpr uint32_t numBins() const
        {
            return size;
        }

        /** Fill histogram
         *
         * @attention This method is synchronizing the worker before returning the handle.
         *
         * @tparam T_Worker
         * @tparam T_Histogram
         * @param worker
         * @param electronHistogram
         * @param volumeScalingFactor
         */
        template<typename T_Worker, typename T_Histogram>
        HDINLINE void fill(
            T_Worker const& worker,
            T_Histogram const& electronHistogram,
            float_X const volumeScalingFactor)
        {
            auto forEachElement = lockstep::makeForEach<T_size>(worker);
            forEachElement(
                [&](uint32_t const idx)
                {
                    energy[idx] = electronHistogram.getBinEnergy(idx);
                    // eV
                    float_X const binWithValue = electronHistogram.getBinWidth(idx);
                    binWidth[idx] = binWithValue;
                    // 1/(sim.unit.length()^3 * eV)
                    density[idx] = electronHistogram.getBinWeight0(idx) / volumeScalingFactor / binWithValue;
                });
            worker.sync();
        }
    };


} // namespace picongpu::particles::atomicPhysics::kernel
