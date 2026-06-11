/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/dimensions/DataSpace.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace pmacc::device
{
    /** Get the number of threads within a block
     *
     * @param acc alpaka accelerator
     */
    template<typename T_Acc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getBlockSize(T_Acc const& acc)
    {
        auto alpakaBlockExtent = ::alpaka::getWorkDiv<::alpaka::Block, ::alpaka::Threads>(acc);
        constexpr uint32_t dim = ::alpaka::Dim<decltype(alpakaBlockExtent)>::value;
        return DataSpace<dim>(alpakaBlockExtent);
    }

    /** Get the number of blocks within a grid
     *
     * @param acc alpaka accelerator
     */
    template<typename T_Acc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getGridSize(T_Acc const& acc)
    {
        auto alpakaGridExtent = ::alpaka::getWorkDiv<::alpaka::Grid, ::alpaka::Blocks>(acc);
        constexpr uint32_t dim = ::alpaka::Dim<decltype(alpakaGridExtent)>::value;
        return DataSpace<dim>(alpakaGridExtent);
    }

    /** Get the thread index within a block
     *
     * @param acc alpaka accelerator
     */
    template<typename T_Acc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getThreadIdx(T_Acc const& acc)
    {
        auto alpakaThreadIdx = ::alpaka::getIdx<::alpaka::Block, ::alpaka::Threads>(acc);
        constexpr uint32_t dim = ::alpaka::Dim<decltype(alpakaThreadIdx)>::value;
        return DataSpace<dim>(alpakaThreadIdx);
    }

    /** Get the block index within a grid
     *
     * @param acc alpaka accelerator
     */
    template<typename T_Acc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getBlockIdx(T_Acc const& acc)
    {
        auto alpakaBlockdIdx = ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Blocks>(acc);
        constexpr uint32_t dim = ::alpaka::Dim<decltype(alpakaBlockdIdx)>::value;
        return DataSpace<dim>(alpakaBlockdIdx);
    }
} // namespace pmacc::device
