/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/types.hpp>

#include <cstdlib>
#include <iostream>

namespace picongpu
{
    struct WindowHeader
    {
        using Size2D = pmacc::DataSpace<DIM2>;

        Size2D size;
        Size2D offset;

        void writeToConsole(std::ostream& ocons) const
        {
            ocons << "WindowHeader.size " << size.x() << " " << size.y() << std::endl;
            ocons << "WindowHeader.offset " << offset.x() << " " << offset.y() << std::endl;
        }
    };

} // namespace picongpu
