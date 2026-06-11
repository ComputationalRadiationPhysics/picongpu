/*
 * SPDX-FileCopyrightText: Felix Schmitt, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/types.hpp"

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
        HINLINE static DataSpace<DIM> getGridDim(Base const& base, uint32_t exchangeType)
        {
            return base.getGridSuperCells();
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getSuperCellIndex(
            Base const& base,
            DataSpace<DIM> const& _blockIdx,
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
        HINLINE static DataSpace<DIM> getGridDim(Base const& base, uint32_t exchangeType)
        {
            DataSpace<DIM> const guardingSupercells = base.getGuardingSuperCells();
            DataSpace<DIM> result(base.getGridSuperCells() - 2 * guardingSupercells);

            DataSpace<DIM> const directions = Mask::getRelativeDirections<DIM>(exchangeType);

            for(uint32_t d = 0; d < DIM; ++d)
            {
                if(directions[d] != 0)
                    result[d] = guardingSupercells[d];
            }

            return result;
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getSuperCellIndex(
            Base const& base,
            DataSpace<DIM> const& _blockIdx,
            uint32_t exchangeType)
        {
            DataSpace<DIM> result(_blockIdx);

            DataSpace<DIM> const directions = Mask::getRelativeDirections<DIM>(exchangeType);
            DataSpace<DIM> const guardingSupercells = base.getGuardingSuperCells();

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
        HINLINE static DataSpace<DIM> getGridDim(Base const& base, uint32_t exchangeType)
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
        HDINLINE static DataSpace<DIM> getSuperCellIndex(
            Base const& base,
            DataSpace<DIM> const& _blockIdx,
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
