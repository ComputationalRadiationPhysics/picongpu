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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/plugins/output/images/PngCreator.hpp"
#include "picongpu/plugins/output/header/MessageHeader.hpp"

#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/verify.hpp>
#include <pmacc/types.hpp>

#include <boost/core/ignore_unused.hpp>

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

#if(PIC_ENABLE_PNG == 1)
#    include <pngwriter.h>
#endif

namespace picongpu
{
    template<class Box>
    inline void PngCreator::createImage(const Box data, const MessageHeader::Size2D size, const MessageHeader header)
    {
#if(PIC_ENABLE_PNG == 1)
        if(m_createFolder)
        {
            Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(m_folder);
            m_createFolder = false;
        }

        std::stringstream step;
        step << std::setw(6) << std::setfill('0') << header.sim.step;
        std::string filename(m_name + "_" + step.str() + ".png");

        pngwriter png(size.x(), size.y(), 0, filename.c_str());

        /* default compression: 6
         * zlib level 1 is ~12% bigger but ~2.3x faster in write_png( )
         */
        png.setcompressionlevel(1);

        // PngWriter coordinate system begin with 1,1
        for(int y = 0; y < size.y(); ++y)
        {
            for(int x = 0; x < size.x(); ++x)
            {
                float3_X p = data[y][x];
                png.plot(x + 1, size.y() - y, p.x(), p.y(), p.z());
            }
        }

        /* scale the image by a user defined relative factor
         * `scale_image` is defined in `png.param`
         */
        float_X scale_x(scale_image);
        float_X scale_y(scale_image);


        if(scale_to_cellsize)
        {
            // scale to real cell size
            scale_x *= header.sim.scale[0];
            scale_y *= header.sim.scale[1];
        }

        /* to prevent artifacts scale only, if at least one of scale_x and
         * scale_y is != 1.0
         */
        if((scale_x != float_X(1.0)) || (scale_y != float_X(1.0)))
            // process the cell size and by factor scaling within one step
            png.scale_kxky(scale_x, scale_y);

        // add some meta information
        // header.writeToConsole( std::cout );

        std::ostringstream description(std::ostringstream::out);
        header.writeToConsole(description);

        char title[] = "PIConGPU preview image";
        std::string author = Environment<>::get().SimulationDescription().getAuthor();
        char software[] = "PIConGPU with PNGwriter";

        png.settext(title, author.c_str(), description.str().c_str(), software);

        // write to disk and close object
        png.close();
#else
        boost::ignore_unused(data, size, header);
        /* always fail with an exception at runtime */
        PMACC_VERIFY_MSG(false, "not allowed to call createImage (missing dependency PNGwriter)");
#endif
    }

} /* namespace picongpu */
