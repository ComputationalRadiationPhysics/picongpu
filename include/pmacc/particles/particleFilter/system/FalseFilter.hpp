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
    class FalseFilter
    {
    public:
        FalseFilter()
        {
        }

        virtual ~FalseFilter()
        {
        }

        template<class T_Particle>
        bool operator()(T_Particle const& particle)
        {
            return false;
        }
    };

} // namespace pmacc
