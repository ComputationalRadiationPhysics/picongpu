/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Common.hpp>

#include <vector>
#include <string>

namespace alpaka
{
    namespace acc
    {
        //#############################################################################
        //! The acceleration properties on a device.
        //
        // \TODO:
        //  TIdx m_maxClockFrequencyHz;            //!< Maximum clock frequency of the device in Hz.
        template<
            typename TDim,
            typename TIdx>
        struct AccDevProps
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST AccDevProps(
                TIdx const & multiProcessorCount,
                vec::Vec<TDim, TIdx> const & gridBlockExtentMax,
                TIdx const & gridBlockCountMax,
                vec::Vec<TDim, TIdx> const & blockThreadExtentMax,
                TIdx const & blockThreadCountMax,
                vec::Vec<TDim, TIdx> const & threadElemExtentMax,
                TIdx const & threadElemCountMax,
                size_t const & sharedMemSizeBytes) :
                    m_gridBlockExtentMax(gridBlockExtentMax),
                    m_blockThreadExtentMax(blockThreadExtentMax),
                    m_threadElemExtentMax(threadElemExtentMax),
                    m_gridBlockCountMax(gridBlockCountMax),
                    m_blockThreadCountMax(blockThreadCountMax),
                    m_threadElemCountMax(threadElemCountMax),
                    m_multiProcessorCount(multiProcessorCount),
                    m_sharedMemSizeBytes(sharedMemSizeBytes)
            {}

            // NOTE: The members have been reordered from the order in the constructor because gcc is buggy for some TDim and TIdx and generates invalid assembly.
            vec::Vec<TDim, TIdx> m_gridBlockExtentMax;      //!< The maximum number of blocks in each dimension of the grid.
            vec::Vec<TDim, TIdx> m_blockThreadExtentMax;    //!< The maximum number of threads in each dimension of a block.
            vec::Vec<TDim, TIdx> m_threadElemExtentMax;     //!< The maximum number of elements in each dimension of a thread.

            TIdx m_gridBlockCountMax;                  //!< The maximum number of blocks in a grid.
            TIdx m_blockThreadCountMax;                //!< The maximum number of threads in a block.
            TIdx m_threadElemCountMax;                 //!< The maximum number of elements in a threads.

            TIdx m_multiProcessorCount;                //!< The number of multiprocessors.
            size_t m_sharedMemSizeBytes;               //!< The size of shared memory per block
        };
    }
}
