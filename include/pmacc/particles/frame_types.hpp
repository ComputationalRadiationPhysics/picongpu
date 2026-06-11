/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
#define INV_IDX (vint_t(0xFFFF'FFFF))

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
