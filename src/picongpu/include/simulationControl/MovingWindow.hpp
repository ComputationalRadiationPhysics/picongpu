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

namespace picongpu
{
using namespace PMacc;

/**
 * Structure holding domain information for this GPU.
 * Current domain information can be obtained using MovingWindow class.
 */
struct DomainInformation
{
    /* Offset from simulation origin to moving window */
    DataSpace<simDim> globalDomainOffset;
    /* Total size of current simulation area (i.e. moving window size) */
    DataSpace<simDim> globalDomainSize;

    /* Offset from simulation origin to this GPU */
    DataSpace<simDim> domainOffset;
    /* Size of this GPU */
    DataSpace<simDim> domainSize;
    
    /* Offset of simulation area (i.e. moving window) from start of this GPU.
     * >= 0 for top GPUs, 0 otherwise */
    DataSpace<simDim> localDomainOffset;

};

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

    /** create a virtual window which descripe local and global offsets and local size which is important
     *  for domain calculations to dump subvolumes of the full computing domain
     * 
     * @param currentStep simulation step
     * @return description of virtual window
     */
    VirtualWindow getVirtualWindow(uint32_t currentStep)
    {

        VirtualWindow window(slideCounter);
        DataSpace<simDim> gridSize(Environment<simDim>::get().SubGrid().getSimulationBox().getLocalSize());
        DataSpace<simDim> globalWindowSize(simSize);
        globalWindowSize.y() -= gridSize.y() * slidingWindowActive;

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

            Mask comm_mask = Environment<simDim>::get().GridController().getCommunicationMask();

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
    
    /**
     * Get domain information for the currently active simulation grid
     * 
     * @param currentStep current simulation step
     * @return active grid domain information
     */
    DomainInformation getActiveDomain(uint32_t currentStep)
    {
        DomainInformation domInfo;
        VirtualWindow window = getVirtualWindow(currentStep);

        /* set global offset (from physical origin) to our first gpu data area */
        domInfo.localDomainOffset = window.localOffset;
        domInfo.globalDomainOffset = window.globalSimulationOffset;
        domInfo.globalDomainSize = window.globalWindowSize;
        domInfo.domainOffset = Environment<simDim>::get().SubGrid().getSimulationBox().getGlobalOffset();

        /* change only the offset of the first gpu
         * localDomainOffset is only non zero for the gpus on top
         */
        domInfo.domainOffset += domInfo.localDomainOffset;
        domInfo.domainSize = window.localSize;
        
        return domInfo;
    }
    
    /**
     * Get domain information for the ghost part of the simulation grid
     * 
     * @param currentStep current simulation step
     * @return ghost grid domain information
     */
    DomainInformation getGhostDomain(uint32_t currentStep)
    {
        DomainInformation domInfo = getActiveDomain(currentStep);
        VirtualWindow window = getVirtualWindow(currentStep);
        
        domInfo.globalDomainOffset.y() += domInfo.globalDomainSize.y();
        domInfo.domainOffset.y() = domInfo.globalDomainOffset.y();
        domInfo.domainSize = window.localFullSize;
        domInfo.domainSize.y() -= window.localSize.y();
        domInfo.globalDomainSize = window.globalSimulationSize;
        domInfo.globalDomainSize.y() -= domInfo.globalDomainOffset.y();
        domInfo.localDomainOffset = DataSpace<simDim > ();
        /* only important for bottom gpus */
        domInfo.localDomainOffset.y() = window.localSize.y();
        
        if (window.isBottom == false)
        {
            /* set size for all gpu to zero which are not bottom gpus */
            domInfo.domainSize.y() = 0;
        }
        
        return domInfo;
    }

};

} //namespace picongpu

