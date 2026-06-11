/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/math/Vector.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
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

        using AddFirst = typename math::CT::add<OffsetOrigin, SuperCellSize>::type;
        using FullSuperCellSize = typename math::CT::add<AddFirst, OffsetEnd>::type;
    };

} // namespace pmacc
