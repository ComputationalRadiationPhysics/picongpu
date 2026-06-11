/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/plugins/output/header/MessageHeader.hpp"
#include "picongpu/plugins/output/images/PngCreator.hpp"
#include "picongpu/plugins/output/images/param.hpp"

#include <pmacc/mappings/simulation/Filesystem.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/types.hpp>
#include <pmacc/verify.hpp>

#include <boost/core/ignore_unused.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if (PIC_ENABLE_PNG == 1)
#    include <pngwriter.h>
#endif

namespace picongpu
{
    template<typename T_DataType>
    inline void PngCreator::createImage(
        std::shared_ptr<std::vector<T_DataType>> imageVector,
        MessageHeader const header)
    {
#if (PIC_ENABLE_PNG == 1)
        if(m_createFolder)
        {
            pmacc::Filesystem::get().createDirectoryWithPermissions(m_folder);
            m_createFolder = false;
        }

        std::stringstream step;
        step << std::setw(6) << std::setfill('0') << header.simHeader.step;
        std::string filename(m_name + "_" + step.str() + ".png");

        auto size = header.window.size;

        pngwriter png(size.x(), size.y(), 0, filename.c_str());

        /* default compression: 6
         * zlib level 1 is ~12% bigger but ~2.3x faster in write_png( )
         */
        png.setcompressionlevel(1);

        auto& img = *imageVector.get();

        // PngWriter coordinate system begin with 1,1
        for(int y = 0; y < size.y(); ++y)
        {
            for(int x = 0; x < size.x(); ++x)
            {
                auto srcIdx = pmacc::math::linearize(size, DataSpace<DIM2>(x, y));
                float3_X p = img[srcIdx];
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
            scale_x *= header.simHeader.scale[0];
            scale_y *= header.simHeader.scale[1];
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
        boost::ignore_unused(imageVector, header);
        /* always fail with an exception at runtime */
        PMACC_VERIFY_MSG(false, "not allowed to call createImage (missing dependency PNGwriter)");
#endif
    }

} /* namespace picongpu */
