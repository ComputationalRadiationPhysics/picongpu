/**
 * Copyright 2013, 2015 Heiko Burau, Rene Widera, Benjamin Worpitz
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
#include "particles/memory/boxes/TileDataBox.hpp"
#include "particles/memory/boxes/PopDataBox.hpp"

#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/PitchedBox.hpp"

namespace PMacc
{


template<class TYPE, class VALUE, unsigned DIM>
class ExchangePopDataBox : public DataBox<PitchedBox<VALUE, DIM1> >
{
public:
    typedef ExchangeMemoryIndex<TYPE, DIM> PopType;

    HDINLINE ExchangePopDataBox(VALUE *data,
                               PopDataBox<TYPE, PopType > virtualMemory) :
    DataBox<PitchedBox<VALUE, DIM1> >(PitchedBox<VALUE, DIM1>(data, DataSpace<DIM1>())),
    virtualMemory(virtualMemory)
    {

    }

    HDINLINE TileDataBox<VALUE> pop(DataSpace<DIM> &superCell)
    {

        TileDataBox<PopType> tmp = virtualMemory.popN(1);
        if (tmp.getSize() == 0)
        {
            return TileDataBox<VALUE > (NULL);
        }
        superCell = tmp[0].getSuperCell();
        return TileDataBox<VALUE > (this->fixedPointer,
                                    DataSpace<DIM1 > (tmp[0].getStartIndex()),
                                    tmp[0].getCount());
    }



protected:
    PMACC_ALIGN8(virtualMemory, PopDataBox<TYPE, PopType >);
};

}
