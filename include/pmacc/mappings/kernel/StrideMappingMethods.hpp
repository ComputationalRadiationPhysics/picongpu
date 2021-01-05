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

namespace pmacc
{
    /**
     * Helper class for StrideMapping.
     * Provides methods called by StrideMapping using template specialization.
     *
     * @tparam areaType area to map to
     * @tparam DIM dimension for mapping (1-3)
     */
    template<uint32_t areaType, unsigned DIM>
    class StrideMappingMethods;

    // CORE + BORDER + GUARD

    template<unsigned DIM>
    class StrideMappingMethods<CORE + BORDER + GUARD, DIM>
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base& base)
        {
            return base.getGridSuperCells();
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> shift(const Base& base, const DataSpace<DIM>& value)
        {
            return value;
        }
    };

    // CORE

    template<unsigned DIM>
    class StrideMappingMethods<CORE, DIM>
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base& base)
        {
            // skip 2 x (border + guard) == 4 x guard
            return base.getGridSuperCells() - 4 * base.getGuardingSuperCells();
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> shift(const Base& base, const DataSpace<DIM>& value)
        {
            // skip guard + border == 2 x guard
            return value + 2 * base.getGuardingSuperCells();
        }
    };

    // CORE+BORDER

    template<unsigned DIM>
    class StrideMappingMethods<CORE + BORDER, DIM>
    {
    public:
        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base& base)
        {
            return base.getGridSuperCells() - 2 * base.getGuardingSuperCells();
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> shift(const Base& base, const DataSpace<DIM>& value)
        {
            return value + base.getGuardingSuperCells();
        }
    };

} // namespace pmacc
