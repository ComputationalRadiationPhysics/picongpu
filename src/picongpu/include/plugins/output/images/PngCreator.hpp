/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include <string>
#include "mappings/simulation/GridController.hpp"

#include <pngwriter.h>

#include <iostream>
#include <sstream>

#include <iomanip>

#include "memory/boxes/PitchedBox.hpp"
#include "memory/boxes/DataBox.hpp"
#include "plugins/output/header/MessageHeader.hpp"

//c includes
#include "sys/stat.h"

namespace picongpu
{
    using namespace PMacc;


    struct PngCreator
    {

        PngCreator(std::string name, std::string folder) : m_name(folder + "/" + name), m_folder(folder), m_createFolder(true)
        {
        }

        static std::string getName()
        {
            return std::string("png");
        }

        ~PngCreator()
        {
        }

        template<class Box>
        void operator()(
                        const Box data,
                        const Size2D size,
                        const MessageHeader & header);

    private:

        void resizeAndScaleImage(pngwriter* png, float_64 scaleFactor)
        {
            if (scaleFactor != 1.)
                png->scale_k(scaleFactor);
        }

        std::string m_name;
        std::string m_folder;
        bool m_createFolder;

    };

    template<>
    inline void PngCreator::operator() < DataBox<PitchedBox<float3_X, DIM2 > > >(
                                                                               const DataBox<PitchedBox<float3_X, DIM2 > > data,
                                                                               const Size2D size,
                                                                               const MessageHeader& header
                                                                               )
    {
        if (m_createFolder)
        {
            Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(m_folder);
            m_createFolder = false;
        }

        std::stringstream step;
        step << std::setw(6) << std::setfill('0') << header.sim.step;
        float_X scale_x = header.sim.scale[0];
        float_X scale_y = header.sim.scale[1];
        std::string filename(m_name + "_" + step.str() + ".png");

        pngwriter png(size.x(), size.y(), 0, filename.c_str());

        /* default compression: 6
         * zlib level 1 is ~12% bigger but ~2.3x faster in write_png()
         */
        png.setcompressionlevel(1);

        //PngWriter coordinate system begin with 1,1
        for (int y = 0; y < size.y(); ++y)
        {
            for (int x = 0; x < size.x(); ++x)
            {
                float3_X p = data[y ][x ];
                png.plot(x + 1, size.y() - y, p.x(), p.y(), p.z());
            }
        }

        // scale to real cell size
        // but, to prevent artifacts:
        //   scale only, if at least one of
        //   scale_x and scale_y is != 1.0
        if (scale_to_cellsize)
            if ((scale_x != float_X(1.0)) || (scale_y != float_X(1.0)))
                png.scale_kxky(scale_x, scale_y);

        // global rescales to save disk space
        resizeAndScaleImage(&png, scale_image);

        // add some meta information
        //header.writeToConsole( std::cout );

        std::ostringstream description( std::ostringstream::out );
        header.writeToConsole( description );

        char title[] = "PIConGPU preview image";
        char author[] = "The awesome PIConGPU-Team";
        char software[] = "PIConGPU with PNGwriter";

        png.settext( title, author, description.str().c_str(), software);

        // write to disk and close object
        png.close();
    }

} /* namespace picongpu */
