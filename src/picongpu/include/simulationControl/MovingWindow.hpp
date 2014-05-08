/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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

#include "simulationControl/VirtualWindow.hpp"
#include "simulationControl/SelectionInformation.hpp"

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

    /**
     * Set the global simulation size (in cells)
     *
     * @param size global simulation size
     */
    void setGlobalSimSize(const DataSpace<simDim> size)
    {
        simSize = size;
    }

    /**
     * Set the number of GPUs in each dimension
     *
     * @param count number of simulating GPUs
     */
    void setGpuCount(const DataSpace<simDim> count)
    {
        gpu = count;
    }

    /**
     * Enable or disable sliding window
     *
     * @param value true to enable, false otherwise
     */
    void setSlidingWindow(bool value)
    {
        slidingWindowActive = value;
    }

    /**
     * Set the number of already performed moving window slides
     *
     * @param slides number of slides
     */
    void setSlideCounter(uint32_t slides)
    {
        slideCounter = slides;
        lastSlideStep = slides;
    }

    /**
     * Returns if sliding window is active
     *
     * @return true if active, false otherwise
     */
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

    /**
     * Create a virtual window which describes local and global offsets and local size
     * which is important for domain calculations to dump subvolumes of the full computing domain
     *
     * @param currentStep current simulation step
     * @return the virtual window
     */
    VirtualWindow getVirtualWindow(uint32_t currentStep)
    {
        DomainInformation domInfo;
        VirtualWindow window(slideCounter);

        window.localDimensions = Selection<simDim>(domInfo.localDomain.size);

        window.globalDimensions = Selection<simDim>(simSize);
        window.globalDimensions.size.y() -= domInfo.localDomain.size.y() * slidingWindowActive;

        if (slidingWindowActive)
        {
            const uint32_t devices = gpu.y();
            const double cell_height = (double) CELL_HEIGHT;
            const double light_way_per_step = ((double) SPEED_OF_LIGHT * (double) DELTA_T);
            double stepsInFuture_tmp = (window.globalDimensions.size.y() * cell_height / light_way_per_step) * (1.0 - slide_point);
            uint32_t stepsInFuture = ceil(stepsInFuture_tmp);
            double stepsInFutureAfterComma = stepsInFuture_tmp - (double) stepsInFuture; /*later used for calculate smoother offsets*/

            /* round to nearest step so we get smaller sliding dfference
             * this is valid if we activate sliding window because y direction has
             * the same size for all gpus
             */
            const uint32_t stepsPerGPU = (uint32_t) math::floor((double) (domInfo.localDomain.size.y() * cell_height) / light_way_per_step + 0.5);
            const uint32_t firstSlideStep = stepsPerGPU * devices - stepsInFuture;
            const uint32_t firstMoveStep = stepsPerGPU * (devices - 1) - stepsInFuture;

            double offsetFirstGPU = 0.0;

            if (firstMoveStep <= currentStep)
            {
                const uint32_t stepsInLastGPU = (currentStep + stepsInFuture) % stepsPerGPU;
                /* moving window start */
                if (firstSlideStep <= currentStep && stepsInLastGPU == 0)
                {
                    window.doSlide = true;
                    if (lastSlideStep != currentStep)
                        slideCounter++;
                    lastSlideStep = currentStep;
                }
                window.slides = slideCounter;

                /* round to nearest cell to have smoother offset jumps */
                offsetFirstGPU = math::floor(((double) stepsInLastGPU + stepsInFutureAfterComma) * light_way_per_step / cell_height + 0.5);

                /* global offset is all 0 except for y dimension */
                window.globalDimensions.offset.y() = offsetFirstGPU;
            }

            Mask comm_mask = Environment<simDim>::get().GridController().getCommunicationMask();

            /* set top/bottom if there are no communication partners
             * for this GPU in the respective direction */
            const bool isTopGpu = !comm_mask.isSet(TOP);
            const bool isBottomGpu = !comm_mask.isSet(BOTTOM);
            window.isTop = isTopGpu;
            window.isBottom = isBottomGpu;

            if (isTopGpu)
            {
                window.localDimensions.offset.y() = offsetFirstGPU;
                window.localDimensions.size.y() -= offsetFirstGPU;
            }
            else if (isBottomGpu)
            {
                window.localDimensions.size.y() = offsetFirstGPU;
            }
        }

        return window;
    }

    /**
     * Return selection information for the currently active simulation grid
     *
     * @param currentStep current simulation timestep
     * @return current domain and window sizes and offsets and active selection
     */
    SelectionInformation getActiveSelection(uint32_t currentStep)
    {
        SelectionInformation sInfo;

        /* gather all required information */
        sInfo.movingWindow = getVirtualWindow(currentStep);

        sInfo.selectionOffset = sInfo.movingWindow.localDimensions.offset;

        sInfo.globalSelection = sInfo.movingWindow.globalDimensions;
        sInfo.localSelection.size = sInfo.movingWindow.localDimensions.size;
        sInfo.localSelection.offset = Environment<simDim>::get().SubGrid().getSimulationBox().getGlobalOffset();

        /* change only the offset of the first gpu
         * localDomainOffset is only non zero for the gpus on top
         */
        sInfo.localSelection.offset += sInfo.selectionOffset;

        return sInfo;
    }

    /**
     * Return selection information for the current ghost part of the simulation grid
     *
     * @param currentStep current simulation step
     * @return current domain and window sizes and offsets and ghost selection
     */
    SelectionInformation getGhostSelection(uint32_t currentStep)
    {
        SelectionInformation sInfo = getActiveSelection(currentStep);

        sInfo.globalSelection.offset.y() += sInfo.globalSelection.size.y();
        sInfo.localSelection.offset.y() = sInfo.globalSelection.offset.y();
        sInfo.localSelection.size = sInfo.domains.localDomain.size;
        sInfo.localSelection.size.y() -= sInfo.movingWindow.localDimensions.size.y();
        sInfo.globalSelection.size = simSize;
        sInfo.globalSelection.size.y() -= sInfo.globalSelection.offset.y();
        sInfo.selectionOffset = DataSpace<simDim > ();

        /* only important for bottom gpus */
        sInfo.selectionOffset.y() = sInfo.movingWindow.localDimensions.size.y();

        if (sInfo.movingWindow.isBottom == false)
        {
            /* set size for all gpu to zero which are not bottom gpus */
            sInfo.localSelection.size.y() = 0;
        }

        return sInfo;
    }

};

} //namespace picongpu

