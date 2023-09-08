/* Copyright 2013-2023 Rene Widera
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/math/Vector.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace ct = pmacc::math::CT;

    /** Define a SuperCell with guarding cells
     *
     * This object describe a SuperCell block that is surrounded by guarding cells
     *
     * @tparam T_SuperCellSize compile time size of a SuperCell per dimension
     * @tparam T_OffsetOrigin compile time size of the guard relative to origin (positive value)
     * @tparam T_OffsetEnd compile time size of the guard relative to end of SuperCell (positive value)
     */
    template<
        class T_SuperCellSize,
        class T_OffsetOrigin = typename math::CT::make_Int<T_SuperCellSize::dim, 0>::type,
        class T_OffsetEnd = typename math::CT::make_Int<T_SuperCellSize::dim, 0>::type>
    struct SuperCellDescription
    {
        enum
        {
            Dim = T_SuperCellSize::dim
        };
        using SuperCellSize = T_SuperCellSize;
        using OffsetOrigin = T_OffsetOrigin;
        using OffsetEnd = T_OffsetEnd;
        using Type = SuperCellDescription<SuperCellSize, OffsetOrigin, OffsetEnd>;

        using AddFirst = typename ct::add<OffsetOrigin, SuperCellSize>::type;
        using FullSuperCellSize = typename ct::add<AddFirst, OffsetEnd>::type;
    };

} // namespace pmacc
