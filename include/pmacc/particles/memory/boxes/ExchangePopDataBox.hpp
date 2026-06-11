/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/PitchedBox.hpp"
#include "pmacc/particles/memory/boxes/TileDataBox.hpp"
#include "pmacc/particles/memory/dataTypes/ExchangeMemoryIndex.hpp"

namespace pmacc
{
    template<class TYPE, class VALUE, unsigned DIM>
    class ExchangePopDataBox : public DataBox<PitchedBox<VALUE, DIM1>>
    {
    public:
        using PopType = ExchangeMemoryIndex<TYPE, DIM>;

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
            return TileDataBox<VALUE>(this->m_ptr, DataSpace<DIM1>(tmp.getStartIndex()), tmp.getCount());
        }

    protected:
        PMACC_ALIGN8(virtualMemory, DataBox<PitchedBox<PopType, DIM1>>);
    };

} // namespace pmacc
