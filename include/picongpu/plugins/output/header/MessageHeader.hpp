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
#include <pmacc/dimensions/DataSpace.hpp>

#include "picongpu/plugins/output/header/DataHeader.hpp"
#include "picongpu/plugins/output/header/NodeHeader.hpp"
#include "picongpu/plugins/output/header/SimHeader.hpp"
#include "picongpu/plugins/output/header/WindowHeader.hpp"

#include "picongpu/simulation/control/Window.hpp"

#include <iostream>
#include <cstdlib>


namespace picongpu
{
    struct MessageHeader
    {
        using Size2D = WindowHeader::Size2D;

        enum
        {
            realBytes = sizeof(DataHeader) + sizeof(SimHeader) + sizeof(WindowHeader) + sizeof(NodeHeader),
            bytes = realBytes < 120 ? 128 : 256
        };

        template<class CellDesc>
        void update(
            CellDesc& cellDesc,
            picongpu::Window vWindow,
            Size2D transpose,
            uint32_t currentStep,
            picongpu::float_32* cellSizeArr = nullptr,
            const pmacc::DataSpace<CellDesc::Dim> gpus = pmacc::DataSpace<CellDesc::Dim>())
        {
            using namespace pmacc;
            using namespace picongpu;

            enum
            {
                Dim = CellDesc::Dim
            };

            const DataSpace<Dim> localSize(cellDesc.getGridLayout().getDataSpaceWithoutGuarding());
            const DataSpace<DIM2> localSize2D(localSize[transpose.x()], localSize[transpose.y()]);

            /*update only if nuber of gpus are set, else use old value*/
            if(gpus.productOfComponents() != 0)
                sim.nodes = DataSpace<DIM2>(gpus[transpose.x()], gpus[transpose.y()]);

            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

            const DataSpace<Dim> globalSize(subGrid.getGlobalDomain().size);
            sim.size.x() = globalSize[transpose.x()];
            sim.size.y() = globalSize[transpose.y()];

            node.maxSize = DataSpace<DIM2>(localSize[transpose.x()], localSize[transpose.y()]);

            const DataSpace<Dim> windowSize = vWindow.globalDimensions.size;
            window.size = DataSpace<DIM2>(windowSize[transpose.x()], windowSize[transpose.y()]);

            if(cellSizeArr != nullptr)
            {
                picongpu::float_32 scale[2];
                scale[0] = cellSizeArr[transpose.x()];
                scale[1] = cellSizeArr[transpose.y()];
                sim.cellSizeArr[0] = cellSizeArr[transpose.x()];
                sim.cellSizeArr[1] = cellSizeArr[transpose.y()];

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
            }

            const DataSpace<Dim> offsetToSimNull(subGrid.getLocalDomain().offset);
            const DataSpace<Dim> windowOffsetToSimNull(vWindow.globalDimensions.offset);
            const DataSpace<Dim> localOffset(vWindow.localDimensions.offset);

            const DataSpace<DIM2> localOffset2D(localOffset[transpose.x()], localOffset[transpose.y()]);
            node.localOffset = localOffset2D;

            DataSpace<Dim> offsetToWindow(offsetToSimNull - windowOffsetToSimNull);

            const DataSpace<DIM2> offsetToWindow2D(offsetToWindow[transpose.x()], offsetToWindow[transpose.y()]);
            node.offsetToWindow = offsetToWindow2D;

            const DataSpace<DIM2> offsetToSimNull2D(offsetToSimNull[transpose.x()], offsetToSimNull[transpose.y()]);
            node.offset = offsetToSimNull2D;

            const DataSpace<DIM2> windowOffsetToSimNull2D(
                windowOffsetToSimNull[transpose.x()],
                windowOffsetToSimNull[transpose.y()]);
            window.offset = windowOffsetToSimNull2D;

            const DataSpace<Dim> currentLocalSize(vWindow.localDimensions.size);
            const DataSpace<DIM2> currentLocalSize2D(currentLocalSize[transpose.x()], currentLocalSize[transpose.y()]);
            node.size = currentLocalSize2D;

            sim.step = currentStep;

            /*add sliding windo informations to header*/
            const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
            sim.simOffsetToNull = DataSpace<DIM2>();
            if(transpose.x() == 1)
                sim.simOffsetToNull.x() = node.maxSize.x() * numSlides;
            else if(transpose.y() == 1)
                sim.simOffsetToNull.y() = node.maxSize.y() * numSlides;
        }

        static MessageHeader* create()
        {
            return (MessageHeader*) new uint8_t[bytes];
        }

        static void destroy(MessageHeader* obj)
        {
            __deleteArray(obj);
        }

        MessageHeader& operator=(MessageHeader const&) = default;

        DataHeader data;
        SimHeader sim;
        WindowHeader window;
        NodeHeader node;
        // ColorHeader color; will be used later on to save channel ranges

        void writeToConsole(std::ostream& ocons) const
        {
            data.writeToConsole(ocons);
            sim.writeToConsole(ocons);
            window.writeToConsole(ocons);
            node.writeToConsole(ocons);
        }

    private:
        /** constructor
         *
         * it is only allowed to create Message header with @see create()
         */
        MessageHeader();
    };

} // namespace picongpu
