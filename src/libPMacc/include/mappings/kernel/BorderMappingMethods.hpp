/**
 * Copyright 2015 Alexander Grund
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

#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "memory/dataTypes/Mask.hpp"

namespace PMacc
{

    /**
     * Helper class for BorderMapping.
     * Provides methods called by BorderMapping using template specialization.
     *
     * @tparam T_Direction direction to map to (TOP; BOTTOM)
     * @tparam DIM dimension for mapping (1-3)
     */
    template<unsigned DIM>
    class BorderMappingMethods;

    // DIM2
    template<>
    class BorderMappingMethods<DIM2>
    {
    public:

        template<class T_Base>
        HINLINE static DataSpace<DIM2>
        getGridDim(const T_Base &base, PMacc::ExchangeType direction)
        {
            DataSpace<DIM2> result(base.getGridSuperCells() - 2 * base.getGuardingSuperCells());

            const DataSpace<DIM2> directions = Mask::getRelativeDirections<DIM2 >(direction);

            if (directions.x() != 0)
                result.x() = base.getBorderSuperCells();
            else if (directions.y() != 0)
                result.y() = base.getBorderSuperCells();

            return result;
        }

        template<class Base>
        HDINLINE static DataSpace<DIM2>
        getBlockOffset(const Base &base, PMacc::ExchangeType direction)
        {
            DataSpace<DIM2> result;

            DataSpace<DIM2> directions = Mask::getRelativeDirections<DIM2 >(direction);

            if (directions.x() == 1)
                result.x() = base.getGridSuperCells().x() - base.getGuardingSuperCells() - base.getBorderSuperCells();
            else
                result.x() = base.getGuardingSuperCells();

            if (directions.y() == 1)
                result.y() = base.getGridSuperCells().y() - base.getGuardingSuperCells() - base.getBorderSuperCells();
            else
                result.y() = base.getGuardingSuperCells();

            return result;
        }
    };

    // DIM3
    template<>
    class BorderMappingMethods<DIM3>
    {
    public:

        template<class T_Base>
        HINLINE static DataSpace<DIM3>
        getGridDim(const T_Base &base, PMacc::ExchangeType direction)
        {
            DataSpace<DIM3> result(base.getGridSuperCells() - 2 * base.getGuardingSuperCells());

            const DataSpace<DIM3> directions = Mask::getRelativeDirections<DIM3 >(direction);

            if (directions.x() != 0)
                result.x() = base.getBorderSuperCells();
            else if (directions.y() != 0)
                result.y() = base.getBorderSuperCells();
            else if (directions.z() != 0)
                result.z() = base.getBorderSuperCells();

            return result;
        }

        template<class Base>
        HDINLINE static DataSpace<DIM3>
        getBlockOffset(const Base &base, PMacc::ExchangeType direction)
        {
            DataSpace<DIM3> result;

            DataSpace<DIM3> directions = Mask::getRelativeDirections<DIM3 >(direction);

            if (directions.x() == 1)
                result.x() = base.getGridSuperCells().x() - base.getGuardingSuperCells() - base.getBorderSuperCells();
            else
                result.x() = base.getGuardingSuperCells();

            if (directions.y() == 1)
                result.y() = base.getGridSuperCells().y() - base.getGuardingSuperCells() - base.getBorderSuperCells();
            else
                result.y() = base.getGuardingSuperCells();

            if (directions.z() == 1)
                result.z() = base.getGridSuperCells().y() - base.getGuardingSuperCells() - base.getBorderSuperCells();
            else
                result.z() = base.getGuardingSuperCells();

            return result;
        }
    };

}//namespace PMacc
