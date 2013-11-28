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
 


#ifndef PARTICLESINITONEPARTICLE_HPP
#define	PARTICLESINITONEPARTICLE_HPP

#include "dimensions/DataSpace.hpp"
#include "types.h"
#include "simulation_classTypes.hpp"
#include "mappings/simulation/SubGrid.hpp"

#include "eventSystem/EventSystem.hpp"
#include "dimensions/DataSpaceOperations.hpp"

#include "plugins/radiation/parameters.hpp"

namespace picongpu
{


template< class ParBox>
__global__ void kernelAddOneParticle(ParBox pb,
                                     DataSpace<DIM3> superCell, DataSpace<DIM3> parLocalCell)
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
        float3_X pos = float3_X(LOCAL_POS_X, LOCAL_POS_Y, LOCAL_POS_Z);

        const float_X GAMMA0 = (float_X) (1.0 / sqrt(1.0 - (BETA0_X * BETA0_X + BETA0_Y * BETA0_Y + BETA0_Z * BETA0_Z)));
        float3_X mom = float3_X(
                                GAMMA0 * par.getMass(parWeighting) * float_X(BETA0_X) * SPEED_OF_LIGHT,
                                GAMMA0 * par.getMass(parWeighting) * float_X(BETA0_Y) * SPEED_OF_LIGHT,
                                GAMMA0 * par.getMass(parWeighting) * float_X(BETA0_Z) * SPEED_OF_LIGHT
                                );

        par[position_] = pos;
        par[momentum_] = mom;
        par[multiMask_] = 1;
        par[localCellIdx_] = linearIdx;
        par[weighting_] = parWeighting;

#if(ENABLE_RADIATION == 1)
        frame[i][momentumPrev1_] = float3_X(0.f, 0.f, 0.f);
        frame[i][radiationFlag_] = true;
#endif
    }
}

template<class ParticlesClass>
class ParticlesInitOneParticle
{
public:

    static void addOneParticle(ParticlesClass& parClass, MappingDesc cellDescription, DataSpace<DIM3> globalCell)
    {

        PMACC_AUTO(simBox, SubGrid<simDim>::getInstance().getSimulationBox());
        const DataSpace<DIM3> globalTopLeft = simBox.getGlobalOffset();
        const DataSpace<DIM3> localSimulationArea = simBox.getLocalSize();
        
        DataSpace<DIM3> localParCell = globalCell - globalTopLeft;


        for (int i = 0; i < (int) DIM3; ++i)
        {
            //chek if particle is in the simulation area
            if (localParCell[i] < 0 || localParCell[i] >= localSimulationArea[i])
                return;
        }

        //calculate supercell 
        DataSpace<DIM3> localSuperCell = (localParCell / MappingDesc::SuperCellSize::getDataSpace());
        DataSpace<DIM3> cellInSuperCell = localParCell - (localSuperCell * MappingDesc::SuperCellSize::getDataSpace());
        //add garding blocks to supercell 
        localSuperCell = localSuperCell + cellDescription.getGuardingSuperCells();

        std::cout << "localParCell: " << localParCell << std::endl;
        std::cout << "localSuperCell: " << localSuperCell << std::endl;
        std::cout << "cellInSuperCell: " << cellInSuperCell << std::endl;

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

#endif	/* PARTICLESINITONEPARTICLE_HPP */

