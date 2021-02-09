/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera
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

#include "pmacc/types.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"

namespace pmacc
{
    /**
     * Helper class for ExchangeMapping.
     * Provides methods called by ExchangeMapping using template specialization.
     *
     * @tparam areaType area to map to (GUARD, BORDER)
     * @tparam DIM dimension for mapping (1-3)
     */
    template<uint32_t areaType, unsigned DIM>
    class ExchangeMappingMethods
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base& base, uint32_t exchangeType)
        {
            return base.getGridSuperCells();
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getBlockIndex(
            const Base& base,
            const DataSpace<DIM>& _blockIdx,
            uint32_t exchangeType)
        {
            return _blockIdx;
        }
    };

    // areaType == GUARD

    template<unsigned DIM>
    class ExchangeMappingMethods<GUARD, DIM>
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base& base, uint32_t exchangeType)
        {
            const DataSpace<DIM> guardingSupercells = base.getGuardingSuperCells();
            DataSpace<DIM> result(base.getGridSuperCells() - 2 * guardingSupercells);

            const DataSpace<DIM> directions = Mask::getRelativeDirections<DIM>(exchangeType);

            for(uint32_t d = 0; d < DIM; ++d)
            {
                if(directions[d] != 0)
                    result[d] = guardingSupercells[d];
            }

            return result;
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getBlockIndex(
            const Base& base,
            const DataSpace<DIM>& _blockIdx,
            uint32_t exchangeType)
        {
            DataSpace<DIM> result(_blockIdx);

            const DataSpace<DIM> directions = Mask::getRelativeDirections<DIM>(exchangeType);
            const DataSpace<DIM> guardingSupercells = base.getGuardingSuperCells();

            for(uint32_t d = 0; d < DIM; ++d)
            {
                if(directions[d] == 0)
                    result[d] += guardingSupercells[d];
                else if(directions[d] == 1)
                    result[d] += base.getGridSuperCells()[d] - guardingSupercells[d];
            }

            return result;
        }
    };


    // areaType == BORDER

    template<unsigned DIM>
    class ExchangeMappingMethods<BORDER, DIM>
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base& base, uint32_t exchangeType)
        {
            // skip 2 x (border + guard) == 4 x guard
            DataSpace<DIM> result(base.getGridSuperCells() - 4 * base.getGuardingSuperCells());

            DataSpace<DIM> directions = Mask::getRelativeDirections<DIM>(exchangeType);

            for(uint32_t d = 0; d < DIM; ++d)
            {
                if(directions[d] != 0)
                    result[d] = base.getGuardingSuperCells()[d];
            }

            return result;
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getBlockIndex(
            const Base& base,
            const DataSpace<DIM>& _blockIdx,
            uint32_t exchangeType)
        {
            DataSpace<DIM> result(_blockIdx);

            DataSpace<DIM> directions = Mask::getRelativeDirections<DIM>(exchangeType);

            DataSpace<DIM> guardingBlocks = base.getGuardingSuperCells();

            for(uint32_t d = 0; d < DIM; ++d)
            {
                switch(directions[d])
                {
                case 0:
                    result[d] += 2 * guardingBlocks[d];
                    break;
                case -1:
                    result[d] += guardingBlocks[d];
                    break;
                case 1:
                    result[d] += base.getGridSuperCells()[d] - 2 * guardingBlocks[d];
                    break;
                }
            }

            return result;
        }
    };

} // namespace pmacc
