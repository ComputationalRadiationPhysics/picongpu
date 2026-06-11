/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/types.hpp>

#include <cstdlib>
#include <iostream>

namespace picongpu
{
    struct SimHeader
    {
        using Size2D = pmacc::DataSpace<2U>;

        Size2D size;
        uint32_t step{0};
        picongpu::float_32 scale[2];
        picongpu::float_32 cellSizeArr[2];

        SimHeader()
        {
            scale[0] = 1.f;
            scale[1] = 1.f;
            cellSizeArr[0] = 0.f;
            cellSizeArr[1] = 0.f;
        }

        void setScale(picongpu::float_32 x, picongpu::float_32 y)
        {
            scale[0] = x;
            scale[1] = y;
        }

        void writeToConsole(std::ostream& ocons) const
        {
            ocons << "SimHeader.size " << size.x() << " " << size.y() << std::endl;
            ocons << "SimHeader.step " << step << std::endl;
            ocons << "SimHeader.scale " << scale[0] << " " << scale[1] << std::endl;
            ocons << "SimHeader.cellSize " << cellSizeArr[0] << " " << cellSizeArr[1] << std::endl;
        }
    };

} // namespace picongpu
