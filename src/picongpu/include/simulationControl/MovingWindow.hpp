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

#include "simulationControl/Window.hpp"

namespace picongpu
{
using namespace PMacc;

/**
 * Singleton class managing the moving window, slides.
 * Can be used to create window views on the grid.
 */
class MovingWindow
{
private:

    MovingWindow() : slidingWindowActive(false), slideCounter(0), lastSlideStep(0)
    {
    }

    MovingWindow(MovingWindow& cc);

    void getCurrentSlideInfo(uint32_t currentStep, bool *doSlide, float_64 *offsetFirstGPU)
    {
        if (doSlide)
            *doSlide = false;

        if (offsetFirstGPU)
            *offsetFirstGPU = 0.0;

        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        const uint32_t windowGlobalDimY =
            subGrid.getGlobalDomain().size.y() - subGrid.getLocalDomain().size.y() * slidingWindowActive;

        const uint32_t devices_y = Environment<simDim>::get().GridController().getGpuNodes().y();
        const float_64 cell_height = (float_64) CELL_HEIGHT;
        const float_64 light_way_per_step = ((float_64) SPEED_OF_LIGHT * (float_64) DELTA_T);
        float_64 stepsInFuture_tmp = (windowGlobalDimY * cell_height / light_way_per_step) * (1.0 - slide_point);
        uint32_t stepsInFuture = ceil(stepsInFuture_tmp);
        /* later used to calculate smoother offsets */
        float_64 stepsInFutureAfterComma = stepsInFuture_tmp - (float_64) stepsInFuture;

        /* round to nearest step so we get smaller sliding dfference
         * this is valid if we activate sliding window because y direction has
         * the same size for all gpus
         */
        const uint32_t stepsPerGPU = (uint32_t) math::floor(
                                                            (float_64) (subGrid.getLocalDomain().size.y() * cell_height) / light_way_per_step + 0.5);
        const uint32_t firstSlideStep = stepsPerGPU * devices_y - stepsInFuture;
        const uint32_t firstMoveStep = stepsPerGPU * (devices_y - 1) - stepsInFuture;

        if (slidingWindowActive==true && firstMoveStep <= currentStep)
        {
            const uint32_t stepsInLastGPU = (currentStep + stepsInFuture) % stepsPerGPU;
            /* moving window start */
            if (firstSlideStep <= currentStep && stepsInLastGPU == 0)
            {
                incrementSlideCounter(currentStep);
                if (doSlide)
                    *doSlide = true;
            }

            /* round to nearest cell to have smoother offset jumps */
            if (offsetFirstGPU)
            {
                *offsetFirstGPU = math::floor(((float_64) stepsInLastGPU + stepsInFutureAfterComma) *
                                              light_way_per_step / cell_height + 0.5);
            }
        }
    }

    /** increment slide counter
     *
     * It is allowed to call this function more than once per time step
     * The function takes care that the counter is only incremented once
     * per simulation step
     *
     * @param current simulation step
     */
    void incrementSlideCounter(const uint32_t currentStep)
    {
        if (slidingWindowActive==true && lastSlideStep != currentStep)
        {
            slideCounter++;
            lastSlideStep = currentStep;
        }
    }

    /** true is sliding window is activated */
    bool slidingWindowActive;

    /** current number of slides since start of simulation */
    uint32_t slideCounter;

    /**
     * last simulation step with slide
     * used to prevent multiple slides per simulation step
     */
    uint32_t lastSlideStep;
public:

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
     * @param currentStep current simulation timestep
     */
    void setSlideCounter(uint32_t slides,uint32_t currentStep)
    {
        slideCounter = slides;
        /* ensure that we will not change the slide counter with `incrementSlideCounter()`
         * in the same time step again
         */
        lastSlideStep = currentStep;
    }

    /**
     * Return the number of slides since start of simulation.
     * If slide occurs in \p currentStep, it is included in the result.
     *
     * @param currentStep current simulation step
     * @return number of slides
     */
    uint32_t getSlideCounter(uint32_t currentStep)
    {
        getCurrentSlideInfo(currentStep, NULL, NULL);
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

    /**
     * Return if a slide occurs in the current simulation step.
     *
     * @param currentStep current simulation step
     * @return true if slide in current step, false otherwise
     */
    bool slideInCurrentStep(uint32_t currentStep)
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
     * Return a window which describes the global and local moving window
     *
     * @param currentStep current simulation step
     * @return moving window
     */
    Window getWindow(uint32_t currentStep)
    {
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        Window window;

        window.localDimensions = Selection<simDim>(subGrid.getLocalDomain().size);
        window.globalDimensions = Selection<simDim>(subGrid.getGlobalDomain().size);

        /* If sliding is inactive, moving window is the same as global domain (substract 0)*/
        window.globalDimensions.size.y() -= subGrid.getLocalDomain().size.y() * slidingWindowActive;

        if (slidingWindowActive)
        {
            float_64 offsetFirstGPU = 0.0;
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
                /* local window offset is relative to global window start */
                window.localDimensions.offset.y() = 0;
                window.localDimensions.size.y() -= offsetFirstGPU;
            }
            else
            {
                window.localDimensions.offset.y() = subGrid.getLocalDomain().offset.y() - offsetFirstGPU;
                if (isBottomGpu)
                {
                    window.localDimensions.size.y() = offsetFirstGPU;
                }
            }
        }

        return window;
    }

    /**
     * Return a window which describes the global and local domain
     *
     * @param currentStep current simulation step
     * @return window over global/local domain
     */
    Window getDomainAsWindow(uint32_t currentStep) const
    {
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        Window window;

        window.localDimensions = subGrid.getLocalDomain();
        window.globalDimensions = subGrid.getGlobalDomain();

        return window;
    }

};

} //namespace picongpu

