/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/particles/frame_types.hpp"
#include "pmacc/particles/memory/frames/NullFrame.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    class TrueFilter
    {
    public:
        HDINLINE TrueFilter() = default;

        template<class T_Particle>
        HDINLINE bool operator()(T_Particle const& particle)
        {
            return true;
        }
    };

} // namespace pmacc
