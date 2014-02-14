/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
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
 


#ifndef LASERPHYSICS_HPP
#define LASERPHYSICS_HPP

#include "types.h"
#include "simulation_defines.hpp"

#include "dimensions/GridLayout.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include <math.h>



namespace picongpu
{
using namespace PMacc;

class LaserManipulator
{
public:

    HINLINE LaserManipulator(float3_X elong, DataSpace<simDim> globalCentered, float_X phase) :
    elong(elong), globalCentered(globalCentered), phase(phase)
    {
        //  std::cout<<globalCenteredXOffset<<std::endl;
    }

    HDINLINE float3_X getManipulation(DataSpace<simDim> iOffset)
    {
        const float_X posX = float_X(globalCentered.x() + iOffset.x()) * CELL_WIDTH;
        
        /*! \todo this is very dirty, please fix laserTransversal interface and use floatD_X
            and not posX,posY */
        const float_X posZ =
#if (SIMDIM==DIM3)
        float_X(globalCentered.z() + iOffset.z()) * CELL_DEPTH;
#else
        0.0;
#endif

        return laserProfile::laserTransversal(elong, phase, posX, posZ);
    }

private:
    float3_X elong;
    float_X phase;
    DataSpace<simDim> globalCentered;
};

class LaserPhysics
{
public:

    LaserPhysics(GridLayout<simDim> layout) :
    layout(layout)
    {
    }

    LaserManipulator getLaserManipulator(uint32_t currentStep)
    {
        float3_X elong;
        float_X phase = 0.0;

        elong = laserProfile::laserLongitudinal(currentStep,
                                                phase);

        SubGrid<simDim>& sg = SubGrid<simDim>::getInstance();
        PMACC_AUTO(simBox,sg.getSimulationBox());
        
        const DataSpace<simDim> globalCellOffset(simBox.getGlobalOffset());
        const DataSpace<simDim> halfSimSize(simBox.getGlobalSize() / 2);
        
        DataSpace<simDim> centeredOrigin(globalCellOffset - halfSimSize);
        return LaserManipulator(elong,
                                centeredOrigin,
                                phase);
    }

private:

    GridLayout<simDim> layout;
};
}

#endif  /* LASERPHYSICS_HPP */

