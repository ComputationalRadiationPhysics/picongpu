/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Tapish Narwal
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/types.hpp>

#include <iomanip>

#include <pngwriter.h>

struct PngCreator
{
    template<class DBox>
    void operator()(uint32_t currentStep, DBox data, pmacc::DataSpace<DIM2> dataSize)
    {
        std::stringstream step;
        step << std::setw(6) << std::setfill('0') << currentStep;
        std::string filename("heat_" + step.str() + ".png");
        pngwriter png(dataSize.x(), dataSize.y(), 0, filename.c_str());
        png.setcompressionlevel(9);

        for(int y = 0; y < dataSize.y(); ++y)
        {
            for(int x = 0; x < dataSize.x(); ++x)
            {
                float p = data[pmacc::DataSpace<DIM2>(x, y)];
                png.plot(x + 1, dataSize.y() - y, p, 0., 1. - p);
            }
        }
        png.close();
    }
};
