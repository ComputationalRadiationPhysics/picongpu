/* Copyright 2013-2021 Felix Schmitt, Rene Widera, Benjamin Worpitz
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

// define which index means that the index is invalid
#define INV_IDX 0xFFFFFFFF

// define which index means that a local cell index is invalid
#define INV_LOC_IDX 0xFFFF

namespace pmacc
{
    /**
     * Is used for indirect pointer layer.
     * This type is limited by atomicSub on device (in CUDA 3.2 we can use 32 Bit int only).
     */
    typedef unsigned int vint_t;

    /**
     * Defines the local cell id type in a supercell
     */
    typedef uint16_t lcellId_t;

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
