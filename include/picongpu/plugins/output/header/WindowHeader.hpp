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
