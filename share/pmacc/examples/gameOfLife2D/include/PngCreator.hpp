/* Copyright 2013-2021 Heiko Burau, Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pmacc/types.hpp>
#include <pngwriter.h>

namespace gol
{
    struct PngCreator
    {
        template<class DBox>
        void operator()(uint32_t currentStep, DBox data, Space dataSize)
        {
            std::stringstream step;
            step << std::setw(6) << std::setfill('0') << currentStep;
            std::string filename("gol_" + step.str() + ".png");
            pngwriter png(dataSize.x(), dataSize.y(), 0, filename.c_str());
            png.setcompressionlevel(9);

            for(int y = 0; y < dataSize.y(); ++y)
            {
                for(int x = 0; x < dataSize.x(); ++x)
                {
                    float p = data[y][x];
                    png.plot(x + 1, dataSize.y() - y, p, p, p);
                }
            }
            png.close();
        }
    };

} // namespace gol
