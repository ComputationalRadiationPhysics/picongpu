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

namespace picongpu
{
using namespace PMacc;

struct VirtualWindow
{

    VirtualWindow() :
    slides(0), doSlide(false), isTop(false), isBottom(false)
    {
    }

    VirtualWindow(uint32_t slides, bool doSlide = false) :
    localSize(SubGrid<simDim>::getInstance().getSimulationBox().getLocalSize()),
    localFullSize(localSize),
    slides(slides),
    doSlide(doSlide)
    {
    }
    /*difference between grid NULL point and first seen data of the simulation*/
    DataSpace<simDim> globalSimulationOffset;

    /*local offset from local NULL point*/
    DataSpace<simDim> localOffset;

    /*new size of local grid which contains the manipulation with the local offset (oldsize-localOffset)*/
    DataSpace<simDim> localSize;

    DataSpace<simDim> globalWindowSize;

    DataSpace<simDim> localFullSize;

    DataSpace<simDim> globalSimulationSize;

    /*slides since begin of the simulation*/
    uint32_t slides;

    /*true if simulation slide in the current round*/
    bool doSlide;

    //only set if sliding window is active
    bool isTop;
    //only set if sliding window is active
    bool isBottom;
};
}

#endif	/* VIRTUALWINDOW_HPP */

