/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/types.hpp>

namespace gol
{
    using namespace pmacc;

    typedef DataSpace<DIM2> Space;
    typedef GridController<DIM2> GC;
    typedef GridBuffer<uint8_t, DIM2> Buffer;

    enum CommunicationTags
    {
        BUFF1 = 0u,
        BUFF2 = 1u
    };
} // namespace gol
