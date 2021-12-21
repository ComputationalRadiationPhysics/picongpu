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
