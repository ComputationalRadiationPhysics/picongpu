/* Copyright 2024 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/vec/Vec.hpp"

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

        // Please keep the order of data members so aggregate initialization does not break!
        TIdx m_multiProcessorCount; //!< The number of multiprocessors.
        Vec<TDim, TIdx> m_gridBlockExtentMax; //!< The maximum number of blocks in each dimension of the grid.
        TIdx m_gridBlockCountMax; //!< The maximum number of blocks in a grid.
        Vec<TDim, TIdx> m_blockThreadExtentMax; //!< The maximum number of threads in each dimension of a block.
        TIdx m_blockThreadCountMax; //!< The maximum number of threads in a block.
        Vec<TDim, TIdx> m_threadElemExtentMax; //!< The maximum number of elements in each dimension of a thread.
        TIdx m_threadElemCountMax; //!< The maximum number of elements in a threads.
        size_t m_sharedMemSizeBytes; //!< The size of shared memory per block
        size_t m_globalMemSizeBytes; //!< The size of global memory
    };
} // namespace alpaka
