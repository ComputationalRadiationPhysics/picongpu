/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/particles/memory/frames/NullFrame.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    class NullFrame
    {
    public:
        enum
        {
            tileSize = 0,
            dim = DIM3
        };
    };

} // namespace pmacc
