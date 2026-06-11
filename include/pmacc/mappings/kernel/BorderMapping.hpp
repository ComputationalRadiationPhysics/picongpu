/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/assert.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/mappings/kernel/MapperConcept.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/types.hpp"

#include <stdexcept>

namespace pmacc
{
    /** Mapping from block indices to supercells in the given border for alpaka kernels
     *
     * Adheres to the MapperConcept.
     *
     * This maps onto the border to 1 exchange direction (e.g. TOP, BOTTOM, TOP + LEFT, ...)
     * Choosing multiple directions defines an intersection [1] in mathematical set theory.
     * The area is basically the same as the surrounding guard region but on the border.
     *
     * Examples:
     * FRONT: Whole top area of the border (Size: ~x*y)
     * FRONT + LEFT: Only the edge at the intersection of the front and left border (Size: ~y)
     * FRONT + LEFT + TOP: Only the corner super cell(s) (Size: ~1)
     *
     * [1] https://en.wikipedia.org/wiki/Intersection_%28set_theory%29
     *
     * @tparam T_BaseClass base class for mapping, should be MappingDescription
     */
    template<class T_BaseClass>
    class BorderMapping;

    template<template<unsigned, class> class T_BaseClass, unsigned T_dim, class T_SuperCellSize>
    class BorderMapping<T_BaseClass<T_dim, T_SuperCellSize>> : public T_BaseClass<T_dim, T_SuperCellSize>
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
        HINLINE BorderMapping(BaseClass const& base, pmacc::ExchangeType direction)
            : BaseClass(base)
            , m_direction(direction)
        {
            PMACC_ASSERT(direction != 0);
        }

        /**
         * Returns the exchange direction used by this mapper
         */
        HDINLINE pmacc::ExchangeType getDirection() const
        {
            return m_direction;
        }

        /** Generate grid dimension information for alpaka kernel calls
         *
         * A kernel using this mapping must use exacly the returned number of blocks
         *
         * @return number of blocks in a grid
         */
        HINLINE DimDataSpace getGridDim() const
        {
            DimDataSpace result(this->getGridSuperCells() - 2 * this->getGuardingSuperCells());

            DimDataSpace const directions = Mask::getRelativeDirections<Dim>(m_direction);

            for(int i = 0; i < Dim; i++)
            {
                if(directions[i] != 0)
                    result[i] = this->getGuardingSuperCells()[i];
            }

            return result;
        }

        /** Return index of a supercell to be processed by the given alpaka block
         *
         * @tparam T_origin Which origin (CORE/BORDER/GUARD) to return supercell index relative to (default: GUARD)
         * @param blockIdx alpaka block index
         * @return mapped SuperCell index including guards
         */
        template<uint32_t T_origin = GUARD>
        HDINLINE DimDataSpace getSuperCellIndex(DimDataSpace const& blockIdx) const
        {
            DimDataSpace result = blockIdx;

            DimDataSpace const directions = Mask::getRelativeDirections<Dim>(m_direction);

            for(int i = 0; i < Dim; i++)
            {
                if(directions[i] == 1)
                    result[i] += this->getGridSuperCells()[i] - 2 * this->getGuardingSuperCells()[i];
                else
                    result[i] += this->getGuardingSuperCells()[i];
            }
            if constexpr(T_origin == CORE)
            {
                result = result - 2 * this->getGuardingSuperCells();
            }
            if constexpr(T_origin == BORDER)
            {
                result = result - this->getGuardingSuperCells();
            }
            return result;
        }

    private:
        PMACC_ALIGN(m_direction, pmacc::ExchangeType const);
    };
} // namespace pmacc
