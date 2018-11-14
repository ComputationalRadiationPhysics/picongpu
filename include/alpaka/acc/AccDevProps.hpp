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
        //  TIdx m_maxClockFrequencyHz;            //!< Maximum clock frequency of the device in Hz.
        //  TIdx m_sharedMemSizeBytes;             //!< Idx of the available block shared memory in bytes.
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
                TIdx const & threadElemCountMax) :
                    m_gridBlockExtentMax(gridBlockExtentMax),
                    m_blockThreadExtentMax(blockThreadExtentMax),
                    m_threadElemExtentMax(threadElemExtentMax),
                    m_gridBlockCountMax(gridBlockCountMax),
                    m_blockThreadCountMax(blockThreadCountMax),
                    m_threadElemCountMax(threadElemCountMax),
                    m_multiProcessorCount(multiProcessorCount)
            {}

            // NOTE: The members have been reordered from the order in the constructor because gcc is buggy for some TDim and TIdx and generates invalid assembly.
            vec::Vec<TDim, TIdx> m_gridBlockExtentMax;      //!< The maximum number of blocks in each dimension of the grid.
            vec::Vec<TDim, TIdx> m_blockThreadExtentMax;    //!< The maximum number of threads in each dimension of a block.
            vec::Vec<TDim, TIdx> m_threadElemExtentMax;     //!< The maximum number of elements in each dimension of a thread.

            TIdx m_gridBlockCountMax;                  //!< The maximum number of blocks in a grid.
            TIdx m_blockThreadCountMax;                //!< The maximum number of threads in a block.
            TIdx m_threadElemCountMax;                 //!< The maximum number of elements in a threads.

            TIdx m_multiProcessorCount;                //!< The number of multiprocessors.
        };
    }
}
