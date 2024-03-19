/* Copyright 2013-2023 Heiko Burau, Rene Widera, Benjamin Worpitz
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/PitchedBox.hpp"
#include "pmacc/particles/memory/boxes/PushDataBox.hpp"
#include "pmacc/particles/memory/boxes/TileDataBox.hpp"
#include "pmacc/particles/memory/dataTypes/ExchangeMemoryIndex.hpp"

namespace pmacc
{
    /**
     * @tparam TYPE type for addresses
     * @tparam VALUE type for actual data
     * @tparam DIM dimension
     */
    template<class TYPE, class VALUE, unsigned DIM>
    class ExchangePushDataBox : public DataBox<PitchedBox<VALUE, DIM1>>
    {
    public:
        using PushType = ExchangeMemoryIndex<TYPE, DIM>;

        /**
         *
         * @param data particle data storage
         * @param particleCount current fill level of data
         * @param maxSize max capacity of data
         * @param indexTable index table which holds meta data to link particles in data to supercells
         */
        HDINLINE ExchangePushDataBox(
            VALUE* data,
            TYPE* particleCount,
            TYPE maxSize,
            PushDataBox<TYPE, PushType> indexTable)
            : DataBox<PitchedBox<VALUE, DIM1>>(PitchedBox<VALUE, DIM1>(data))
            , m_indexTable(indexTable)
            , m_particleCount(particleCount)
            , m_maxSize(maxSize)
        {
        }

        /** give access to push N elements into the memory
         *
         * The method is threadsave within the given alpaka hierarchy.
         *
         * @tparam T_Acc type of the alpaka accelerator
         * @tparam T_Hierarchy alpaka::hierarchy type of the hierarchy
         *
         * @param acc alpaka accelerator
         * @param count number of elements to increase stack with
         * @param superCell offset of the supercell relative to the local domain
         * @param hierarchy alpaka parallelism hierarchy levels guarantee valid
         *                  concurrency access to the memory
         *
         * @return a TileDataBox of size count pointing to the new stack elements
         */
        template<typename T_Worker, typename T_Hierarchy>
        HDINLINE TileDataBox<VALUE> pushN(
            T_Worker const& worker,
            TYPE count,
            DataSpace<DIM> const& superCell,
            T_Hierarchy const& hierarchy)
        {
            // offset in destination array for our particle data
            TYPE oldSize = alpaka::atomicAdd(worker.getAcc(), m_particleCount, count, hierarchy);

            if(oldSize + count > m_maxSize)
            {
                // reset size to maxsize
                alpaka::atomicExch(worker.getAcc(), m_particleCount, m_maxSize, hierarchy);
                if(oldSize >= m_maxSize)
                {
                    return TileDataBox<VALUE>(nullptr, DataSpace<DIM1>(0), 0);
                }
                else
                    count = m_maxSize - oldSize;
            }

            TileDataBox<PushType> tmp = m_indexTable.pushN(worker, 1, hierarchy);
            tmp[0].setSuperCell(superCell);
            tmp[0].setCount(count);
            tmp[0].setStartIndex(oldSize);
            return TileDataBox<VALUE>(this->m_ptr, DataSpace<DIM1>(oldSize), count);
        }


    protected:
        PMACC_ALIGN8(m_indexTable, PushDataBox<TYPE, PushType>);
        PMACC_ALIGN(m_particleCount, TYPE*);
        PMACC_ALIGN(m_maxSize, TYPE);
    };

} // namespace pmacc
