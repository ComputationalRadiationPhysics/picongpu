/*
 * SPDX-FileCopyrightText: Felix Schmitt, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/types.hpp"

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
        HINLINE static DataSpace<DIM> getGridDim(Base const& base)
        {
            return base.getGridSuperCells();
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> shift(Base const& base, DataSpace<DIM> const& value)
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
        HINLINE static DataSpace<DIM> getGridDim(Base const& base)
        {
            // skip 2 x (border + guard) == 4 x guard
            return base.getGridSuperCells() - 4 * base.getGuardingSuperCells();
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> shift(Base const& base, DataSpace<DIM> const& value)
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
        HINLINE static DataSpace<DIM> getGridDim(Base const& base)
        {
            return base.getGridSuperCells() - 2 * base.getGuardingSuperCells();
        }

        template<class Base>
        HDINLINE static DataSpace<DIM> shift(Base const& base, DataSpace<DIM> const& value)
        {
            return value + base.getGuardingSuperCells();
        }
    };

} // namespace pmacc
