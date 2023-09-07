/* Copyright 2013-2023 Felix Schmitt, Rene Widera, Benjamin Worpitz
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

namespace pmacc
{
    /**
     * Is used for indirect pointer layer.
     * This type is limited by atomicSub on device (in CUDA 3.2 we can use 32 Bit int only).
     */
    using vint_t = unsigned int;
    //! define which index means that the index is invalid
#define INV_IDX (vint_t(0xFFFFFFFF))

    /**
     * Defines the local cell id type in a supercell
     */
    using lcellId_t = uint16_t;
    //! define which index means that a local cell index is invalid
#define INV_LOC_IDX (lcellId_t(0xFFFF))

    /**
     * Describes type of a frame (core, border)
     */
    enum FrameType
    {
        CORE_FRAME = 0u,
        BORDER_FRAME = 1u,
        BIG_FRAME = 2u
    };
} // namespace pmacc
