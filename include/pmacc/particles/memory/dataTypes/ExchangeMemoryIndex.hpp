/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    template<class TYPE, unsigned DIM>
    class ExchangeMemoryIndex
    {
    public:
        HDINLINE ExchangeMemoryIndex() : startIdx(0), count(0)
        {
        }

        HDINLINE void setStartIndex(TYPE startIdx)
        {
            this->startIdx = startIdx;
        }

        HDINLINE void setCount(TYPE count)
        {
            this->count = count;
        }

        HDINLINE void setSuperCell(DataSpace<DIM> superCell)
        {
            this->superCell = superCell;
        }

        HDINLINE TYPE getStartIndex()
        {
            return startIdx;
        }

        HDINLINE TYPE getCount()
        {
            return count;
        }

        HDINLINE DataSpace<DIM> getSuperCell()
        {
            return superCell;
        }

    private:
        PMACC_ALIGN(superCell, DataSpace<DIM>);
        PMACC_ALIGN(startIdx, TYPE);
        PMACC_ALIGN(count, TYPE);
    };
} // namespace pmacc
