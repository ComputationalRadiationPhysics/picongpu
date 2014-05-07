/**
 * Copyright 2013 Rene Widera
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
 


#ifndef VIRTUALWINDOW_HPP
#define	VIRTUALWINDOW_HPP

#include "types.h"
#include "mappings/simulation/SubGrid.hpp"
#include "simulationControl/Selection.hpp"

namespace picongpu
{
using namespace PMacc;

struct VirtualWindow
{

    /**
     * Constructor
     * 
     * Initializes number of slides to 0
     */
    VirtualWindow() :
    localDimensions(Environment<simDim>::get().SubGrid().getSimulationBox().getLocalSize()),
    localDomainSize(localDimensions.size),
    slides(0),
    doSlide(false),
    isTop(false),
    isBottom(false)
    {
    }

    /**
     * Constructor
     * 
     * @param slides number of slides since start of simulation
     * @param doSlide
     */
    VirtualWindow(uint32_t slides, bool doSlide = false) :
    localDimensions(Environment<simDim>::get().SubGrid().getSimulationBox().getLocalSize()),
    localDomainSize(localDimensions.size),
    slides(slides),
    doSlide(doSlide),
    isTop(false),
    isBottom(false)
    {
    }

    /* Dimensions (size/offset) of the global virtual window over all GPUs */
    Selection<simDim> globalDimensions;
    
    /* Dimensions (size/offset) of the local virtual window on this GPU */
    Selection<simDim> localDimensions;

    /* domain size of this GPU */
    DataSpace<simDim> localDomainSize;

    /* number of slides since the begin of the simulation */
    uint32_t slides;

    /* true if simulation slide in the current round */
    bool doSlide;

    /* True if this is a 'top' GPU (y position is 0), false otherwise
     * only set if sliding window is active */
    bool isTop;
    
    /* True if this is a 'bottom' GPU (y position is y_size - 1), false otherwise
     * only set if sliding window is active */
    bool isBottom;
};
}

#endif	/* VIRTUALWINDOW_HPP */

