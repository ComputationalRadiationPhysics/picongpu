/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
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


#ifndef STRIDEMAPPINGMETHODS_H
#define	STRIDEMAPPINGMETHODS_H

#include "types.h"
#include "dimensions/DataSpace.hpp"

namespace PMacc
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

    //CORE + BORDER + GUARD

    template<unsigned DIM>
    class StrideMappingMethods<CORE + BORDER + GUARD, DIM>
    {
    public:

        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base &base)
        {
            return base.getGridSuperCells();
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> shift(const Base &base, const DataSpace<DIM>& value)
        {
            return value;
        }
    };

    //CORE

    template<unsigned DIM>
    class StrideMappingMethods<CORE, DIM>
    {
    public:

        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base &base)
        {
            return base.getGridSuperCells() - (2 * (base.getGuardingSuperCells() + base.getBorderSuperCells()));
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> shift(const Base &base, const DataSpace<DIM>& value)
        {
            return value + (base.getGuardingSuperCells() + base.getBorderSuperCells());
        }

    };

    //CORE+BORDER

    template<unsigned DIM>
    class StrideMappingMethods<CORE + BORDER, DIM>
    {
    public:

        template<class Base>
        HINLINE static DataSpace<DIM> getGridDim(const Base &base)
        {
            return base.getGridSuperCells() - (2 * base.getGuardingSuperCells());
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> shift(const Base &base, const DataSpace<DIM>& value)
        {
            return value + base.getGuardingSuperCells();
        }
    };




}


#endif	/* AREAMAPPINGMETHODS_H */

