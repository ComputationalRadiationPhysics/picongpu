/* Copyright 2013-2018 Felix Schmitt, Heiko Burau, Rene Widera
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
        HINLINE static DataSpace<DIM> getGridDim(const Base &base, uint32_t exchangeType)
        {
            return base.getGridSuperCells();
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> getBlockIndex(const Base &base,
        const DataSpace<DIM>& _blockIdx, uint32_t exchangeType)
        {
            return _blockIdx;
        }
    };

    // DIM2

    // areaType == GUARD

    template<>
    class ExchangeMappingMethods<GUARD, DIM2>
    {
    public:

        template<class Base>
        HINLINE static DataSpace<DIM2> getGridDim(const Base &base, uint32_t exchangeType)
        {
            DataSpace<DIM2> result(base.getGridSuperCells() - 2 * base.getGuardingSuperCells());

            DataSpace<DIM2> directions = Mask::getRelativeDirections<DIM2 > (exchangeType);

            for( uint32_t d = 0; d < DIM2; ++d )
            {
                if (directions[ d ] != 0)
                    result[ d ] = base.getGuardingSuperCells()[ d ];
            }

            return result;
        }

        template<class Base>
        HDINLINE static DataSpace<DIM2> getBlockIndex(const Base &base,
        const DataSpace<DIM2>& _blockIdx, uint32_t exchangeType)
        {
            DataSpace<DIM2> result(_blockIdx);

            DataSpace<DIM2> directions = Mask::getRelativeDirections<DIM2 > (exchangeType);

            for( uint32_t d = 0; d < DIM2; ++d )
            {
                if (directions[ d ] == 0)
                    result[ d ] += base.getGuardingSuperCells()[ d ];
                else
                    if (directions[ d ] == 1)
                    result[ d ] += base.getGridSuperCells()[ d ] - base.getGuardingSuperCells()[ d ];
            }

            return result;
        }
    };


    // areaType == BORDER

    template<>
    class ExchangeMappingMethods<BORDER, DIM2>
    {
    public:

        template<class Base>
        HINLINE static DataSpace<DIM2> getGridDim(const Base &base, uint32_t exchangeType)
        {
            DataSpace<DIM2> result(base.getGridSuperCells() - 2 * base.getGuardingSuperCells() -
                    2 * base.getGuardingSuperCells());

            DataSpace<DIM2> directions = Mask::getRelativeDirections<DIM2 > (exchangeType);

            for( uint32_t d = 0; d < DIM2; ++d )
            {
                if (directions[ d ] != 0)
                    result[ d ] = base.getGuardingSuperCells()[ d ];
            }

            return result;
        }

        template<class Base>
        HDINLINE static DataSpace<DIM2> getBlockIndex(const Base &base,
        const DataSpace<DIM2>& _blockIdx, uint32_t exchangeType)
        {
            DataSpace<DIM2> result(_blockIdx);

            DataSpace<DIM2> directions = Mask::getRelativeDirections<DIM2 > (exchangeType);

            DataSpace<DIM2> guardingBlocks = base.getGuardingSuperCells();

            for( uint32_t d = 0; d < DIM2; ++d )
            {
                switch( directions[ d ] )
                {
                    case 0:
                        result[ d ] += guardingBlocks[ d ] + base.getGuardingSuperCells()[ d ];
                        break;
                    case -1:
                        result[ d ] += guardingBlocks[ d ];
                        break;
                    case 1:
                        result[ d ] += base.getGridSuperCells()[ d ] - 2 * guardingBlocks[ d ];
                        break;
                }
            }

            return result;
        }
    };


    // DIM3

    // areaType == GUARD

    template<>
    class ExchangeMappingMethods<GUARD, DIM3>
    {
    public:

        template<class Base>
        HINLINE static DataSpace<DIM3> getGridDim(const Base &base, uint32_t exchangeType)
        {
            const DataSpace<DIM3> guardingSupercells = base.getGuardingSuperCells();
            DataSpace<DIM3> result(base.getGridSuperCells() - 2 * guardingSupercells);

            const DataSpace<DIM3> directions = Mask::getRelativeDirections<DIM3 > (exchangeType);

            for( uint32_t d = 0; d < DIM3; ++d )
            {
                if (directions[ d ] != 0)
                    result[ d ] = guardingSupercells[ d ];
            }

            return result;
        }

        template<class Base>
        HDINLINE static DataSpace<DIM3> getBlockIndex(const Base &base,
        const DataSpace<DIM3>& _blockIdx, uint32_t exchangeType)
        {
            DataSpace<DIM3> result(_blockIdx);

            const DataSpace<DIM3> directions = Mask::getRelativeDirections<DIM3 > (exchangeType);
            const DataSpace<DIM3> guardingSupercells = base.getGuardingSuperCells();

            for( uint32_t d = 0; d < DIM3; ++d )
            {
                if (directions[ d ] == 0)
                    result[ d ] += guardingSupercells[ d ];
                else
                    if (directions[ d ] == 1)
                    result[ d ] += base.getGridSuperCells()[ d ] - guardingSupercells[ d ];
            }

            return result;
        }
    };


    // areaType == BORDER

    template<>
    class ExchangeMappingMethods<BORDER, DIM3>
    {
    public:

        template<class Base>
        HINLINE static DataSpace<DIM3> getGridDim(const Base &base, uint32_t exchangeType)
        {
            DataSpace<DIM3> result(base.getGridSuperCells() - 2 * base.getGuardingSuperCells() -
                    2 * base.getGuardingSuperCells());

            DataSpace<DIM3> directions = Mask::getRelativeDirections<DIM3 > (exchangeType);

            for( uint32_t d = 0; d < DIM3; ++d )
            {
                if (directions[ d ] != 0)
                    result[ d ] = base.getGuardingSuperCells()[ d ];
            }

            return result;
        }

        template<class Base>
        HDINLINE static DataSpace<DIM3> getBlockIndex(const Base &base,
        const DataSpace<DIM3>& _blockIdx, uint32_t exchangeType)
        {
            DataSpace<DIM3> result(_blockIdx);

            DataSpace<DIM3> directions = Mask::getRelativeDirections<DIM3 > (exchangeType);

            DataSpace<DIM3> guardingBlocks = base.getGuardingSuperCells();

            for( uint32_t d = 0; d < DIM3; ++d )
            {
                switch (directions[ d ])
                {
                    case 0:
                        result[ d ] += 2 * guardingBlocks[ d ];
                        break;
                    case -1:
                        result[ d ] += guardingBlocks[ d ];
                        break;
                    case 1:
                        result[ d ] += base.getGridSuperCells()[ d ] - 2 * guardingBlocks[ d ];
                        break;
                }
            }

            return result;
        }
    };

}//namespace pmacc
