/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pmacc/types.hpp>
#include <pmacc/dimensions/DataSpace.hpp>

#include <iostream>
#include <cstdlib>


namespace picongpu
{
    struct SimHeader
    {
        typedef pmacc::DataSpace<DIM2> Size2D;

        Size2D size;
        Size2D nodes;
        Size2D simOffsetToNull;
        uint32_t step;
        picongpu::float_32 scale[2];
        picongpu::float_32 cellSizeArr[2];


        SimHeader() : step(0)
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
            ocons << "SimHeader.nodes " << nodes.x() << " " << nodes.y() << std::endl;
            ocons << "SimHeader.step " << step << std::endl;
            ocons << "SimHeader.scale " << scale[0] << " " << scale[1] << std::endl;
            ocons << "SimHeader.cellSize " << cellSizeArr[0] << " " << cellSizeArr[1] << std::endl;
        }
    };

} // namespace picongpu
