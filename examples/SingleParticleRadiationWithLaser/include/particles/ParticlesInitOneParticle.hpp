/**
 * Copyright 2013 Axel Huebl, Rene Widera, Richard Pausch
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
#include "dimensions/DataSpace.hpp"

#include "mappings/simulation/SubGrid.hpp"

#include "eventSystem/EventSystem.hpp"
#include "dimensions/DataSpaceOperations.hpp"

#include "plugins/radiation/parameters.hpp"

namespace picongpu
{


template< class ParBox>
__global__ void kernelAddOneParticle(ParBox pb,
                                     DataSpace<simDim> superCell, DataSpace<simDim> parLocalCell)
{
    typedef typename ParBox::FrameType FRAME;

    FRAME *frame;

    int linearIdx = DataSpaceOperations<simDim>::template map<MappingDesc::SuperCellSize > (parLocalCell);

    float_X parWeighting = NUM_EL_PER_PARTICLE;

    frame = &(pb.getEmptyFrame());
    pb.setAsLastFrame(*frame, superCell);




    // many particle loop:
    for (unsigned i = 0; i < 1; ++i)
    {
        PMACC_AUTO(par,(*frame)[i]);
        floatD_X pos;
        for(int i=0; i<simDim; ++i)
          pos[i] = 0.5;

        const float_X GAMMA0_X = 1.0f / sqrtf(1.0f - float_X(BETA0_X * BETA0_X));
        const float_X GAMMA0_Y = 1.0f / sqrtf(1.0f - float_X(BETA0_Y * BETA0_Y));
        const float_X GAMMA0_Z = 1.0f / sqrtf(1.0f - float_X(BETA0_Z * BETA0_Z));
        float3_X mom = float3_X(
                                GAMMA0_X * traits::attribute::getMass(parWeighting,par) * float_X(BETA0_X) * SPEED_OF_LIGHT,
                                GAMMA0_Y * traits::attribute::getMass(parWeighting,par) * float_X(BETA0_Y) * SPEED_OF_LIGHT,
                                GAMMA0_Z * traits::attribute::getMass(parWeighting,par) * float_X(BETA0_Z) * SPEED_OF_LIGHT
                                );

        par[position_] = pos;
        par[momentum_] = mom;
        par[multiMask_] = 1;
        par[localCellIdx_] = linearIdx;
        par[weighting_] = parWeighting;

#if(ENABLE_RADIATION == 1)
        par[momentumPrev1_] = float3_X(0.f, 0.f, 0.f);
#if(RAD_MARK_PARTICLE>1) || (RAD_ACTIVATE_GAMMA_FILTER!=0)
        /*this code tree is only passed if we not select any particle*/
        par[radiationFlag_] = true;
#endif
#endif
    }
}

template<class ParticlesClass>
class ParticlesInitOneParticle
{
public:

    static void addOneParticle(ParticlesClass& parClass, MappingDesc cellDescription, DataSpace<simDim> globalCell)
    {

        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        const DataSpace<simDim> globalTopLeft = subGrid.getLocalDomain().offset;
        const DataSpace<simDim> localSimulationArea = subGrid.getLocalDomain().size;
        DataSpace<simDim> localParCell = globalCell - globalTopLeft;


        for (int i = 0; i < (int) simDim; ++i)
        {
            //chek if particle is in the simulation area
            if (localParCell[i] < 0 || localParCell[i] >= localSimulationArea[i])
                return;
        }

        //calculate supercell
        DataSpace<simDim> localSuperCell = (localParCell / MappingDesc::SuperCellSize::toRT());
        DataSpace<simDim> cellInSuperCell = localParCell - (localSuperCell * MappingDesc::SuperCellSize::toRT());
        //add garding blocks to supercell
        localSuperCell = localSuperCell + cellDescription.getGuardingSuperCells();


        __cudaKernel(kernelAddOneParticle)
            (1, 1)
            (parClass.getDeviceParticlesBox(),
             localSuperCell, cellInSuperCell);

        parClass.fillAllGaps();

        std::cout << "Wait for add particle" << std::endl;
        __getTransactionEvent().waitForFinished();
    }
};
}



