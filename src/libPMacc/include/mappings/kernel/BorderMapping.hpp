/**
 * Copyright 2013-2015 Alexander Grund
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
#include <stdexcept>

namespace PMacc
{

    /**
     * This maps onto the border to 1 exchange direction (e.g. TOP, BOTTOM, TOP + LEFT, ...)
     * The area is basically the same as the surrounding guard region
     * but on the border. By "adding" directions you limit the area. Examples:
     * FRONT: Whole top area of the border (Size: ~x*y)
     * FRONT + LEFT: Only the edge at the intersection of the front and left border (Size: ~y)
     * FRONT + LEFT + TOP: Only the corner super cell(s) (Size: ~1)
     *
     * @tparam T_BaseClass base class for mapping, should be MappingDescription
     */
    template<class T_BaseClass>
    class BorderMapping;

    template<
        template<unsigned, class> class T_BaseClass,
        unsigned T_dim,
        class T_SuperCellSize
    >
    class BorderMapping<T_BaseClass<T_dim, T_SuperCellSize> >: public T_BaseClass<T_dim, T_SuperCellSize>
    {
    public:
        typedef T_BaseClass<T_dim, T_SuperCellSize> BaseClass;

        enum
        {
            Dim = BaseClass::Dim,
            AreaType = BORDER
        };
        typedef DataSpace<Dim> DimDataSpace;

        typedef typename BaseClass::SuperCellSize SuperCellSize;

        /**
         * Constructor.
         *
         * @param base object of base class baseClass (see template parameters)
         * @param direction exchange direction to map to
         */
        HINLINE BorderMapping(const BaseClass& base, PMacc::ExchangeType direction): BaseClass(base), m_direction(direction)
        {
            assert(direction != 0);
        }

        /**
         * Returns the exchange direction used by this mapper
         */
        HDINLINE PMacc::ExchangeType
        getDirection() const
        {
            return m_direction;
        }

        /**
         * Generate grid dimension information for kernel calls
         *
         * @return size of the grid
         */
        HINLINE DimDataSpace getGridDim() const
        {
            DimDataSpace result(this->getGridSuperCells() - 2 * this->getGuardingSuperCells());

            const DimDataSpace directions = Mask::getRelativeDirections<Dim>(m_direction);

            for(int i = 0; i < Dim; i++)
            {
                if (directions[i] != 0)
                    result[i] = this->getBorderSuperCells();
            }

            return result;
        }

        /**
         * Returns index of current logical block
         *
         * @param realSuperCellIdx current SuperCell index (block index)
         * @return mapped SuperCell index
         */
        HDINLINE DimDataSpace getSuperCellIndex(const DimDataSpace& realSuperCellIdx) const
        {
            DimDataSpace result = realSuperCellIdx;

            const DimDataSpace directions = Mask::getRelativeDirections<Dim>(m_direction);

            for(int i = 0; i < Dim; i++)
            {
                if (directions[i] == 1)
                    result[i] += this->getGridSuperCells()[i] - this->getGuardingSuperCells() - this->getBorderSuperCells();
                else
                    result[i] += this->getGuardingSuperCells();
            }

            return result;
        }
    private:
        PMACC_ALIGN(m_direction, const PMacc::ExchangeType);
    };
} // namespace PMacc
