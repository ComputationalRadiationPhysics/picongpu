/**
 * Copyright 2013-2014 Rene Widera
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

#pragma once

#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "math/vector/compile-time/Vector.hpp"

namespace PMacc
{
    namespace ct = PMacc::math::CT;

    /** Define a SuperCell with guarding cells
     *
     * This object describe a SuperCell block that is surrounded by guarding cells
     *
     * @tparam T_SuperCellSize compile time size of a SuperCell per dimension
     * @tparam T_OffsetOrigin compile time size of the guard relative to origin (positive value)
     * @tparam T_OffsetEnd compile time size of the guard relative to end of SuperCell (positive value)
     */
    template< class T_SuperCellSize,
    class T_OffsetOrigin = typename math::CT::make_Int<T_SuperCellSize::dim, 0>::type,
    class T_OffsetEnd = typename math::CT::make_Int<T_SuperCellSize::dim, 0>::type >
    struct SuperCellDescription
    {

        enum
        {
            Dim = T_SuperCellSize::dim
        };
        typedef T_SuperCellSize SuperCellSize;
        typedef T_OffsetOrigin OffsetOrigin;
        typedef T_OffsetEnd OffsetEnd;
        typedef SuperCellDescription<SuperCellSize, OffsetOrigin, OffsetEnd> Type;

        typedef typename ct::add<OffsetOrigin,SuperCellSize>::type AddFirst;
        typedef typename ct::add<AddFirst,OffsetEnd>::type FullSuperCellSize;
    };

}//namespace
