/* Copyright 2013-2021 Axel Huebl, Rene Widera
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
#include <string>


namespace picongpu
{
    /** Color Header for Preview Images
     *
     *  Used to store the relation of color channels to min/max units
     *  and data names they represent.
     */
    struct ColorHeader
    {
        struct channel
        {
            /// assign a physical meaningful name to the channel
            std::string name;
            /// assign a unit to the range values
            std::string unitName;
            /// min/max real values for 0 and 255
            picongpu::float_32 range[2];
        };

        channel particles;
        channel channel1;
        channel channel2;
        channel channel3;

        ColorHeader()
        {
            particles.range[0] = 0.f;
            particles.range[1] = 0.f;

            channel1.range[0] = 0.f;
            channel1.range[1] = 0.f;

            channel2.range[0] = 0.f;
            channel2.range[1] = 0.f;

            channel3.range[0] = 0.f;
            channel3.range[1] = 0.f;
        }

        // void setScale(picongpu::float_32 x, picongpu::float_32 y)
        //{
        //    scale[0] = x;
        //    scale[1] = y;
        //}

        void writeToConsole(std::ostream& ocons) const
        {
            // ocons << "ColorHeader.XYZ " << "..." << std::endl;
        }
    };

} // namespace picongpu
