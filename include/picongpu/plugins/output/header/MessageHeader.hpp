/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera
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

#include "picongpu/plugins/output/header/NodeHeader.hpp"
#include "picongpu/plugins/output/header/SimHeader.hpp"
#include "picongpu/plugins/output/header/WindowHeader.hpp"
#include "picongpu/simulation/control/Window.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

#include <cstdlib>
#include <iostream>


namespace picongpu
{
    struct MessageHeader
    {
        using Size2D = WindowHeader::Size2D;

        MessageHeader(picongpu::Window vWindow, Size2D transpose, uint32_t currentStep)
        {
            using namespace pmacc;
            using namespace picongpu;

            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

            const auto localSize(subGrid.getLocalDomain().size);
            const Size2D localSize2D(localSize[transpose.x()], localSize[transpose.y()]);

            const auto globalSize(subGrid.getGlobalDomain().size);
            sim.size.x() = globalSize[transpose.x()];
            sim.size.y() = globalSize[transpose.y()];

            node.maxSize = Size2D(localSize[transpose.x()], localSize[transpose.y()]);

            const auto windowSize = vWindow.globalDimensions.size;
            window.size = Size2D(windowSize[transpose.x()], windowSize[transpose.y()]);

            picongpu::float_32 scale[2];
            scale[0] = cellSize[transpose.x()];
            scale[1] = cellSize[transpose.y()];
            sim.cellSizeArr[0] = cellSize[transpose.x()];
            sim.cellSizeArr[1] = cellSize[transpose.y()];

            const picongpu::float_32 scale0to1 = scale[0] / scale[1];

            if(scale0to1 > 1.0f)
            {
                sim.setScale(scale0to1, 1.f);
            }
            else if(scale0to1 < 1.0f)
            {
                sim.setScale(1.f, 1.0f / scale0to1);
            }
            else
            {
                sim.setScale(1.f, 1.f);
            }

            const auto offsetToSimNull(subGrid.getLocalDomain().offset);
            const auto windowOffsetToSimNull(vWindow.globalDimensions.offset);

            const Size2D offsetToSimNull2D(offsetToSimNull[transpose.x()], offsetToSimNull[transpose.y()]);
            node.offset = offsetToSimNull2D;

            const Size2D windowOffsetToSimNull2D(
                windowOffsetToSimNull[transpose.x()],
                windowOffsetToSimNull[transpose.y()]);
            window.offset = windowOffsetToSimNull2D;

            const auto currentLocalSize(vWindow.localDimensions.size);
            const Size2D currentLocalSize2D(currentLocalSize[transpose.x()], currentLocalSize[transpose.y()]);
            node.size = currentLocalSize2D;

            sim.step = currentStep;
        }

        MessageHeader& operator=(MessageHeader const&) = default;

        SimHeader sim;
        WindowHeader window;
        NodeHeader node;

        void writeToConsole(std::ostream& ocons) const
        {
            sim.writeToConsole(ocons);
            window.writeToConsole(ocons);
            node.writeToConsole(ocons);
        }
    };

} // namespace picongpu
