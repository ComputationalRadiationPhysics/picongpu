/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2013-2024 Axel Huebl, Heiko Burau, Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/defines.hpp"
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

            SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();

            auto const localSize(subGrid.getLocalDomain().size);
            Size2D const localSize2D(localSize[transpose.x()], localSize[transpose.y()]);

            auto const globalSize(subGrid.getGlobalDomain().size);
            simHeader.size.x() = globalSize[transpose.x()];
            simHeader.size.y() = globalSize[transpose.y()];

            node.maxSize = Size2D(localSize[transpose.x()], localSize[transpose.y()]);

            auto const windowSize = vWindow.globalDimensions.size;
            window.size = Size2D(windowSize[transpose.x()], windowSize[transpose.y()]);

            picongpu::float_32 scale[2];
            scale[0] = sim.pic.getCellSize()[transpose.x()];
            scale[1] = sim.pic.getCellSize()[transpose.y()];
            simHeader.cellSizeArr[0] = sim.pic.getCellSize()[transpose.x()];
            simHeader.cellSizeArr[1] = sim.pic.getCellSize()[transpose.y()];

            picongpu::float_32 const scale0to1 = scale[0] / scale[1];

            if(scale0to1 > 1.0f)
            {
                simHeader.setScale(scale0to1, 1.f);
            }
            else if(scale0to1 < 1.0f)
            {
                simHeader.setScale(1.f, 1.0f / scale0to1);
            }
            else
            {
                simHeader.setScale(1.f, 1.f);
            }

            auto const offsetToSimNull(subGrid.getLocalDomain().offset);
            auto const windowOffsetToSimNull(vWindow.globalDimensions.offset);

            Size2D const offsetToSimNull2D(offsetToSimNull[transpose.x()], offsetToSimNull[transpose.y()]);
            node.offset = offsetToSimNull2D;

            Size2D const windowOffsetToSimNull2D(
                windowOffsetToSimNull[transpose.x()],
                windowOffsetToSimNull[transpose.y()]);
            window.offset = windowOffsetToSimNull2D;

            auto const currentLocalSize(vWindow.localDimensions.size);
            Size2D const currentLocalSize2D(currentLocalSize[transpose.x()], currentLocalSize[transpose.y()]);
            node.size = currentLocalSize2D;

            simHeader.step = currentStep;
        }

        MessageHeader& operator=(MessageHeader const&) = default;

        SimHeader simHeader;
        WindowHeader window;
        NodeHeader node;

        void writeToConsole(std::ostream& ocons) const
        {
            simHeader.writeToConsole(ocons);
            window.writeToConsole(ocons);
            node.writeToConsole(ocons);
        }
    };

} // namespace picongpu
