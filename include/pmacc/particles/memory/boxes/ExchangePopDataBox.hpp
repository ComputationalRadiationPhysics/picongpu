/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/particles/memory/dataTypes/ExchangeMemoryIndex.hpp"
#include "pmacc/particles/memory/boxes/TileDataBox.hpp"

#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/PitchedBox.hpp"

namespace pmacc
{
    template<class TYPE, class VALUE, unsigned DIM>
    class ExchangePopDataBox : public DataBox<PitchedBox<VALUE, DIM1>>
    {
    public:
        typedef ExchangeMemoryIndex<TYPE, DIM> PopType;

        HDINLINE ExchangePopDataBox(
            DataBox<PitchedBox<VALUE, DIM1>> data,
            DataBox<PitchedBox<PopType, DIM1>> virtualMemory)
            : DataBox<PitchedBox<VALUE, DIM1>>(data)
            , virtualMemory(virtualMemory)
        {
        }

        HDINLINE
        TileDataBox<VALUE> get(TYPE idx, DataSpace<DIM>& superCell)
        {
            PopType tmp = virtualMemory[idx];

            superCell = tmp.getSuperCell();
            return TileDataBox<VALUE>(this->fixedPointer, DataSpace<DIM1>(tmp.getStartIndex()), tmp.getCount());
        }

    protected:
        PMACC_ALIGN8(virtualMemory, DataBox<PitchedBox<PopType, DIM1>>);
    };

} // namespace pmacc
