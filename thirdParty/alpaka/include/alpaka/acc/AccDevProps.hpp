/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
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
        //  TSize m_maxClockFrequencyHz;            //!< Maximum clock frequency of the device in Hz.
        //  TSize m_sharedMemSizeBytes;             //!< Size of the available block shared memory in bytes.
        template<
            typename TDim,
            typename TSize>
        struct AccDevProps
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST AccDevProps(
                TSize const & multiProcessorCount,
                vec::Vec<TDim, TSize> const & gridBlockExtentMax,
                TSize const & gridBlockCountMax,
                vec::Vec<TDim, TSize> const & blockThreadExtentMax,
                TSize const & blockThreadCountMax,
                vec::Vec<TDim, TSize> const & threadElemExtentMax,
                TSize const & threadElemCountMax) :
                    m_gridBlockExtentMax(gridBlockExtentMax),
                    m_blockThreadExtentMax(blockThreadExtentMax),
                    m_threadElemExtentMax(threadElemExtentMax),
                    m_gridBlockCountMax(gridBlockCountMax),
                    m_blockThreadCountMax(blockThreadCountMax),
                    m_threadElemCountMax(threadElemCountMax),
                    m_multiProcessorCount(multiProcessorCount)
            {}

            // NOTE: The members have been reordered from the order in the constructor because gcc is buggy for some TDim and TSize and generates invalid assembly.
            vec::Vec<TDim, TSize> m_gridBlockExtentMax;      //!< The maximum number of blocks in each dimension of the grid.
            vec::Vec<TDim, TSize> m_blockThreadExtentMax;    //!< The maximum number of threads in each dimension of a block.
            vec::Vec<TDim, TSize> m_threadElemExtentMax;     //!< The maximum number of elements in each dimension of a thread.

            TSize m_gridBlockCountMax;                  //!< The maximum number of blocks in a grid.
            TSize m_blockThreadCountMax;                //!< The maximum number of threads in a block.
            TSize m_threadElemCountMax;                 //!< The maximum number of elements in a threads.

            TSize m_multiProcessorCount;                //!< The number of multiprocessors.
        };
    }
}
