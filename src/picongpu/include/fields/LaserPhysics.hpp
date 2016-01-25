/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
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

#include "simulation_defines.hpp"
#include "fields/LaserPhysics.def"
#include "dimensions/GridLayout.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include <cmath>



namespace picongpu
{
using namespace PMacc;

class LaserManipulator
{
public:

    HINLINE LaserManipulator(float3_X elong, DataSpace<simDim> globalCentered, float_X phase) :
    m_elong(elong), m_globalCentered(globalCentered), m_phase(phase)
    {
        //  std::cout<<globalCenteredXOffset<<std::endl;
    }

    HDINLINE float3_X getManipulation(DataSpace<simDim> iOffset)
    {
        const float_X posX = float_X(m_globalCentered.x() + iOffset.x()) * CELL_WIDTH;

        /*! \todo this is very dirty, please fix laserTransversal interface and use floatD_X
            and not posX,posY */
        const float_X posZ =
#if (SIMDIM==DIM3)
        float_X(m_globalCentered.z() + iOffset.z()) * CELL_DEPTH;
#else
        0.0;
#endif

        return laserProfile::laserTransversal(m_elong, m_phase, posX, posZ);
    }

private:
    float3_X m_elong;
    float_X m_phase;
    DataSpace<simDim> m_globalCentered;
};

class LaserPhysics
{
public:

    LaserPhysics(GridLayout<simDim> layout) :
    m_layout(layout)
    {
    }

    LaserManipulator getLaserManipulator(uint32_t currentStep)
    {
        float3_X elong;
        float_X phase = 0.0;

        elong = laserProfile::laserLongitudinal(currentStep,
                                                phase);

        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

        const DataSpace<simDim> globalCellOffset(subGrid.getLocalDomain().offset);
        const DataSpace<simDim> halfSimSize(subGrid.getGlobalDomain().size / 2);

        DataSpace<simDim> centeredOrigin(globalCellOffset - halfSimSize);
        return LaserManipulator(elong,
                                centeredOrigin,
                                phase);
    }

private:

    GridLayout<simDim> m_layout;
};
}
