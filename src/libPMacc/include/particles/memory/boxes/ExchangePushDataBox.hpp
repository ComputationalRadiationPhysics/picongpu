/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera, Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "particles/memory/dataTypes/ExchangeMemoryIndex.hpp"
#include "memory/boxes/DataBox.hpp"
#include "particles/memory/boxes/TileDataBox.hpp"
#include "particles/memory/boxes/PushDataBox.hpp"

#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/PitchedBox.hpp"

namespace PMacc
{


/**
 * @tparam TYPE type for addresses
 * @tparam VALUE type for actual data
 * @tparam DIM dimension
 */
template<class TYPE, class VALUE, unsigned DIM>
class ExchangePushDataBox : public DataBox<PitchedBox<VALUE, DIM1> >
{
public:

    typedef ExchangeMemoryIndex<TYPE, DIM> PushType;

    HDINLINE ExchangePushDataBox(VALUE *data, TYPE *currentSizePointer, TYPE maxSize,
                                PushDataBox<TYPE, PushType > virtualMemory) :
    DataBox<PitchedBox<VALUE, DIM1> >(PitchedBox<VALUE, DIM1>(data, DataSpace<DIM1>())),
    currentSizePointer(currentSizePointer),
    maxSize(maxSize),
    virtualMemory(virtualMemory)
    {
    }

    HDINLINE TileDataBox<VALUE> pushN(TYPE count, const DataSpace<DIM> &superCell)
    {
        TYPE oldSize = atomicAdd(currentSizePointer, count); //get count VALUEs

        if (oldSize + count > maxSize)
        {
            atomicExch(currentSizePointer, maxSize); //reset size to maxsize
            if (oldSize >= maxSize)
            {
                return TileDataBox<VALUE > (NULL,
                                            DataSpace<DIM1 > (0),
                                            0);
            }
            else
                count = maxSize - oldSize;
        }

        TileDataBox<PushType> tmp = virtualMemory.pushN(1);
        tmp[0].setSuperCell(superCell);
        tmp[0].setCount(count);
        tmp[0].setStartIndex(oldSize);
        return TileDataBox<VALUE > (this->fixedPointer,
                                    DataSpace<DIM1 > (oldSize),
                                    count);
    }



protected:
    PMACC_ALIGN8(virtualMemory, PushDataBox<TYPE, PushType >);
    PMACC_ALIGN(maxSize, TYPE);
    PMACC_ALIGN(currentSizePointer, TYPE*);
};

}
