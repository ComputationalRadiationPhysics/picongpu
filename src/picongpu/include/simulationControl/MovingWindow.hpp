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



#ifndef MOVINGWINDOW_HPP
#define	MOVINGWINDOW_HPP

#include "types.h"
#include "simulation_defines.hpp"

#include "simulationControl/VirtualWindow.hpp"

namespace picongpu
{
using namespace PMacc;

class MovingWindow
{
private:

    MovingWindow() : slidingWindowActive(false), slideCounter(0), lastSlideStep(0)
    {
    }

    MovingWindow(MovingWindow& cc);


    DataSpace<simDim> simSize;
    DataSpace<simDim> gpu;
    bool slidingWindowActive;
    uint32_t slideCounter;
    uint32_t lastSlideStep;
public:

    void setGlobalSimSize(DataSpace<simDim> size)
    {
        simSize = size;
    }

    void setGpuCount(DataSpace<simDim> count)
    {
        gpu = count;
    }

    void setSlidingWindow(bool value)
    {
        slidingWindowActive = value;
    }

    void setSlideCounter(uint32_t slides)
    {
        slideCounter = slides;
        lastSlideStep=slides;
    }

    bool isSlidingWindowActive()
    {
        return slidingWindowActive;
    }

    /**
     * Returns an instance of MovingWindow
     *
     * @return an instance
     */
    static MovingWindow& getInstance()
    {
        static MovingWindow instance;
        return instance;
    }

    /** create a virtual window which descripe local and global offsets and local size which is impotant
     *  for domain calculations to dump subvolumes of the full computing domain
     * 
     * @param currentStep simulation step
     * @return description of virtual window
     */
    VirtualWindow getVirtualWindow(const uint32_t currentStep)
    {

        VirtualWindow window(slideCounter);
        DataSpace<simDim> gridSize(SubGrid<simDim>::getInstance().getSimulationBox().getLocalSize());
        DataSpace<simDim> globalWindowSize(simSize);
        globalWindowSize.y()-=gridSize.y() * slidingWindowActive;

        window.globalWindowSize = globalWindowSize;
        window.globalSimulationSize = simSize;

        if (this->slidingWindowActive)
        {
            const uint32_t devices = gpu.y();
            const double cell_height = (double) CELL_HEIGHT;
            const double light_way_per_step = ((double) SPEED_OF_LIGHT * (double) DELTA_T);
            double stepsInFuture_tmp = (globalWindowSize.y() * cell_height / light_way_per_step) * (1.0 - slide_point);
            uint32_t stepsInFuture = ceil(stepsInFuture_tmp);
            double stepsInFutureAfterComma = stepsInFuture_tmp - (double) stepsInFuture; /*later used for calculate smoother offsets*/

            /*round to nearest step thus we get smaller sliding differgenze
             * this is valid if we activate sliding window because y direction has same size for all gpus
             */
            const uint32_t stepsPerGPU = (uint32_t) math::floor((double) (gridSize.y() * cell_height) / light_way_per_step + 0.5);
            const uint32_t firstSlideStep = stepsPerGPU * devices - stepsInFuture;
            const uint32_t firstMoveStep = stepsPerGPU * (devices - 1) - stepsInFuture;

            double offsetFirstGPU = 0.0;

            if (firstMoveStep <= currentStep)
            {
                const uint32_t stepsInLastGPU = (currentStep + stepsInFuture) % stepsPerGPU;
                //moveing window start
                if (firstSlideStep <= currentStep && stepsInLastGPU == 0)
                {
                    window.doSlide = true;
                    if (lastSlideStep != currentStep)
                        slideCounter++;
                    lastSlideStep = currentStep;
                }
                window.slides = slideCounter;

                /*round to nearest cell to have smoother offset jumps*/
                offsetFirstGPU = math::floor(((double) stepsInLastGPU + stepsInFutureAfterComma) * light_way_per_step / cell_height + 0.5);

                window.globalSimulationOffset.y() = offsetFirstGPU;
            }

            Mask comm_mask = GridController<simDim>::getInstance().getCommunicationMask();

            const bool isTopGpu = !comm_mask.isSet(TOP);
            const bool isBottomGpu = !comm_mask.isSet(BOTTOM);
            window.isTop = isTopGpu;
            window.isBottom = isBottomGpu;

            if (isTopGpu)
            {
                //  std::cout << "----\nstep " << currentStep << " firstMove " << firstMoveStep << std::endl;
                ///  std::cout << "firstSlide" << firstSlideStep << " steps in last " << (double) ((uint32_t) (currentStep + stepsInFuture) % (uint32_t) stepsPerGPU) << std::endl;
                //   std::cout << "Top off: " << offsetFirstGPU << " size " << window.localSize.y() - offsetFirstGPU << std::endl;
                window.localOffset.y() = offsetFirstGPU;
                window.localSize.y() -= offsetFirstGPU;
            }
            else if (isBottomGpu)
            {
                //std::cout<<"Bo  size "<<offsetFirstGPU<<std::endl;
                window.localSize.y() = offsetFirstGPU;
            }


        }

        return window;
    }

};

} //namespace

#endif	/* MOVINGWINDOW_HPP */

