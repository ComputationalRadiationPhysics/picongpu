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

#include "simulationControl/DomainInformation.hpp"
#include "simulationControl/Window.hpp"
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
    
    void getCurrentSlideInfo(uint32_t currentStep, bool *doSlide, double *offsetFirstGPU) const
    {
        if (doSlide)
            *doSlide = false;
        
        if (offsetFirstGPU)
            *offsetFirstGPU = 0.0;
        
        DomainInformation domInfo;
        const uint32_t windowGlobalDimY = simSize.y() - domInfo.localDomain.size.y() * slidingWindowActive;
        
        const uint32_t devices = gpu.y();
        const double cell_height = (double) CELL_HEIGHT;
        const double light_way_per_step = ((double) SPEED_OF_LIGHT * (double) DELTA_T);
        double stepsInFuture_tmp = (windowGlobalDimY * cell_height / light_way_per_step) * (1.0 - slide_point);
        uint32_t stepsInFuture = ceil(stepsInFuture_tmp);
        /* later used to calculate smoother offsets */
        double stepsInFutureAfterComma = stepsInFuture_tmp - (double) stepsInFuture; 

        /* round to nearest step so we get smaller sliding dfference
         * this is valid if we activate sliding window because y direction has
         * the same size for all gpus
         */
        const uint32_t stepsPerGPU = (uint32_t) math::floor(
            (double) (domInfo.localDomain.size.y() * cell_height) / light_way_per_step + 0.5);
        const uint32_t firstSlideStep = stepsPerGPU * devices - stepsInFuture;
        const uint32_t firstMoveStep = stepsPerGPU * (devices - 1) - stepsInFuture;

        if (firstMoveStep <= currentStep)
        {
            const uint32_t stepsInLastGPU = (currentStep + stepsInFuture) % stepsPerGPU;
            /* moving window start */
            if (firstSlideStep <= currentStep && stepsInLastGPU == 0)
            {
                if (doSlide)
                    *doSlide = true;
            }

            /* round to nearest cell to have smoother offset jumps */
            if (offsetFirstGPU)
            {
                *offsetFirstGPU = math::floor(((double) stepsInLastGPU + stepsInFutureAfterComma) *
                        light_way_per_step / cell_height + 0.5);
            }
        }
    }


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
    
    uint32_t getSlideCounter(uint32_t currentStep)
    {
        bool doSlide = false;
        getCurrentSlideInfo(currentStep, &doSlide, NULL);

        if (doSlide && (lastSlideStep != currentStep))
        {
            slideCounter++;
            lastSlideStep = currentStep;
        }
        
        return slideCounter;
    }

    /**
     * Returns if sliding window is active
     *
     * @return true if active, false otherwise
     */
    bool isSlidingWindowActive() const
    {
        return slidingWindowActive;
    }
    
    bool slideInCurrentStep(uint32_t currentStep) const
    {
        bool doSlide = false;
        
        if (slidingWindowActive)
        {
            getCurrentSlideInfo(currentStep, &doSlide, NULL);
        }
        
        return doSlide;
    }
    
    /** 
     * Return true if this is a 'bottom' GPU (y position is y_size - 1), false otherwise
     * only set if sliding window is active
     */
    bool isBottomGPU(void) const
    {
        Mask comm_mask = Environment<simDim>::get().GridController().getCommunicationMask();
        return !comm_mask.isSet(BOTTOM);
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
     * Create a window which describes local and global offsets
     * and sizes of the moving window.
     *
     * @param currentStep current simulation step
     * @return moving window
     */
    Window getWindow(uint32_t currentStep) const
    {
        DomainInformation domInfo;
        Window window;

        window.localDimensions = Selection<simDim>(domInfo.localDomain.size);

        window.globalDimensions = Selection<simDim>(simSize);
        window.globalDimensions.size.y() -= domInfo.localDomain.size.y() * slidingWindowActive;

        if (slidingWindowActive)
        {
            double offsetFirstGPU = 0.0;
            getCurrentSlideInfo(currentStep, NULL, &offsetFirstGPU);

            /* global offset is all 0 except for y dimension */
            window.globalDimensions.offset.y() = offsetFirstGPU;

            /* set top/bottom if there are no communication partners
             * for this GPU in the respective direction */
            const Mask comm_mask = Environment<simDim>::get().GridController().getCommunicationMask();
            const bool isTopGpu = !comm_mask.isSet(TOP);
            const bool isBottomGpu = !comm_mask.isSet(BOTTOM);

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
        sInfo.movingWindow = getWindow(currentStep);

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

        bool isBottomGpu = false;
        
        if (slidingWindowActive)
        {
            const Mask comm_mask = Environment<simDim>::get().GridController().getCommunicationMask();
            isBottomGpu = !comm_mask.isSet(BOTTOM);
        }
        
        if (isBottomGpu == false)
        {
            /* set size for all gpu to zero which are not bottom gpus */
            sInfo.localSelection.size.y() = 0;
        }

        return sInfo;
    }

};

} //namespace picongpu

