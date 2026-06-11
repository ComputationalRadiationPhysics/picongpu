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
    struct NodeHeader
    {
        using Size2D = pmacc::DataSpace<2U>;

        Size2D maxSize;
        Size2D size;
        Size2D offset;

        void writeToConsole(std::ostream& ocons) const
        {
            ocons << "NodeHeader.maxSize " << maxSize.x() << " " << maxSize.y() << std::endl;
            ocons << "NodeHeader.size " << size.x() << " " << size.y() << std::endl;
            ocons << "NodeHeader.offset " << offset.x() << " " << offset.y() << std::endl;
        }
    };

} // namespace picongpu
