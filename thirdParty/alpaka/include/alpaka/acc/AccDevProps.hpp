/* Copyright 2020 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/vec/Vec.hpp"

#include <string>
#include <vector>

namespace alpaka
{
    //! The acceleration properties on a device.
    //
    // \TODO:
    //  TIdx m_maxClockFrequencyHz;            //!< Maximum clock frequency of the device in Hz.
    template<typename TDim, typename TIdx>
    struct AccDevProps
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

        ALPAKA_FN_HOST AccDevProps(
            TIdx const& multiProcessorCount,
            Vec<TDim, TIdx> const& gridBlockExtentMax,
            TIdx const& gridBlockCountMax,
            Vec<TDim, TIdx> const& blockThreadExtentMax,
            TIdx const& blockThreadCountMax,
            Vec<TDim, TIdx> const& threadElemExtentMax,
            TIdx const& threadElemCountMax,
            size_t const& sharedMemSizeBytes)
            : m_gridBlockExtentMax(gridBlockExtentMax)
            , m_blockThreadExtentMax(blockThreadExtentMax)
            , m_threadElemExtentMax(threadElemExtentMax)
            , m_gridBlockCountMax(gridBlockCountMax)
            , m_blockThreadCountMax(blockThreadCountMax)
            , m_threadElemCountMax(threadElemCountMax)
            , m_multiProcessorCount(multiProcessorCount)
            , m_sharedMemSizeBytes(sharedMemSizeBytes)
        {
        }

        // NOTE: The members have been reordered from the order in the constructor because gcc is buggy for some TDim
        // and TIdx and generates invalid assembly.
        Vec<TDim, TIdx> m_gridBlockExtentMax; //!< The maximum number of blocks in each dimension of the grid.
        Vec<TDim, TIdx> m_blockThreadExtentMax; //!< The maximum number of threads in each dimension of a block.
        Vec<TDim, TIdx> m_threadElemExtentMax; //!< The maximum number of elements in each dimension of a thread.

        TIdx m_gridBlockCountMax; //!< The maximum number of blocks in a grid.
        TIdx m_blockThreadCountMax; //!< The maximum number of threads in a block.
        TIdx m_threadElemCountMax; //!< The maximum number of elements in a threads.

        TIdx m_multiProcessorCount; //!< The number of multiprocessors.
        size_t m_sharedMemSizeBytes; //!< The size of shared memory per block
    };
} // namespace alpaka
