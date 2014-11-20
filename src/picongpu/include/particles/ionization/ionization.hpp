/**
 * Copyright 2014 Marco Garten
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


/* includes from Particles.tpp */
#include <iostream>

#include "simulation_defines.hpp"
#include "particles/Particles.hpp" 
#include <cassert>


#include "particles/Particles.kernel"

#include "dataManagement/DataConnector.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"
#include "fields/FieldTmp.hpp"

#include "particles/memory/buffers/ParticlesBuffer.hpp"
#include "particles/ParticlesInit.kernel"
#include "mappings/simulation/GridController.hpp"
#include "mpi/SeedPerRank.hpp"

#include "simulationControl/MovingWindow.hpp"

#include <assert.h>
#include <limits>

#include "fields/numericalCellTypes/YeeCell.hpp"

//#include "particles/traits/GetIonizer.hpp"

/* includes from Particles.kernel */
#include "types.h"
#include "particles/frame_types.hpp"
#include "particles/memory/boxes/ParticlesBox.hpp"
#include "particles/memory/boxes/TileDataBox.hpp"
#include "simulation_types.hpp"
#include "simulation_defines.hpp"

#include "fields/FieldE.hpp"
#include "fields/FieldB.hpp"

#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/CachedBox.hpp"

#include <curand_kernel.h>

#include "nvidia/functors/Assign.hpp"
#include "mappings/threads/ThreadCollective.hpp"

#include "plugins/radiation/parameters.hpp"

#include "nvidia/rng/RNG.hpp"
#include "nvidia/rng/methods/Xor.hpp"
#include "nvidia/rng/distributions/Normal_float.hpp"

#include "particles/operations/Assign.hpp"
#include "particles/operations/Deselect.hpp"

/* includes from particlePusherFree */
#include "types.h"

#include "particles/ionization/ionizationMethods.hpp"

namespace picongpu
{
    
using namespace PMacc;

    

/* IONIZE PER FRAME (formerly PUSH PER FRAME from Particles.kernel) */

template<class IonizeAlgo, class TVec, class T_Field2ParticleInterpolation, class NumericalCellType>
struct IonizeParticlesPerFrame
{

    template<class FrameType, class BoxB, class BoxE >
    DINLINE void operator()(FrameType& ionFrame, int localIdx, BoxB& bBox, BoxE& eBox, int& newElectrons)
    {

        typedef TVec Block;
        /* @new type for field to particle interpolation */
        typedef T_Field2ParticleInterpolation Field2ParticleInterpolation;

        typedef typename BoxB::ValueType BType;
        typedef typename BoxE::ValueType EType;

        PMACC_AUTO(particle,ionFrame[localIdx]);
        const float_X weighting = particle[weighting_];
        
        floatD_X pos = particle[position_];
        const int particleCellIdx = particle[localCellIdx_];

        DataSpace<TVec::dim> localCell(DataSpaceOperations<TVec::dim>::template map<TVec > (particleCellIdx));

       
        EType eField = Field2ParticleInterpolation()
            (eBox.shift(localCell).toCursor(), pos, NumericalCellType::getEFieldPosition());
        BType bField =Field2ParticleInterpolation()
            (bBox.shift(localCell).toCursor(), pos, NumericalCellType::getBFieldPosition());

        float3_X mom = particle[momentum_];
        const float_X mass = getMass<FrameType>(weighting);
        
        /*define charge state variable*/
        int chState = particle[chargeState_];
        
        IonizeAlgo ionizer;
        ionizer(
             bField, eField,
             pos,
             mom,
             mass,
             getCharge<FrameType>(weighting, chState),
             chState
             );
        
        particle[momentum_] = mom;

        particle[position_] = pos;
        
        /* determine number of new electrons to be created */
        newElectrons = chState - particle[chargeState_];
        /* retrieve new charge state */
        particle[chargeState_] = chState;

        /*calculate one dimensional cell index*/
        particle[localCellIdx_] = DataSpaceOperations<TVec::dim>::template map<TVec > (localCell);


        //particle[multiMask_] = direction;

    }
};

/* MARK PARTICLE (formerly MOVE AND MARK from Particles.kernel) */

/*template<class BlockDescription_, class ParBox, class BBox, class EBox, class Mapping, class FrameIonizer>*/
template<class BlockDescription_, class IONFRAME, class ELECTRONFRAME, class BBox, class EBox, class Mapping, class FrameIonizer>
__global__ void kernelIonizeParticles(ParticlesBox<IONFRAME, simDim> ionBox,
                                      ParticlesBox<ELECTRONFRAME, simDim> electronBox,
                                      EBox fieldE,
                                      BBox fieldB,
                                      FrameIonizer frameIonizer,
                                      Mapping mapper)
{
    /* for not mixing assign up with the nVidia functor assign */
    namespace partOp = PMacc::particles::operations;
    
    /* definitions for domain variables, like indices of blocks and threads
     *  
     * conversion from block to linear frames */
    typedef typename BlockDescription_::SuperCellSize SuperCellSize;
    /* "offset" 3D distance to origin in units of super cells */
    const DataSpace<simDim> block(mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx)));

    /* 3D vector from origin of the block to a cell in units of cells */
    const DataSpace<simDim > threadIndex(threadIdx);
    /* conversion from a 3D cell coordinate to a linear coordinate of the cell in its super cell */
    const int linearThreadIdx = DataSpaceOperations<simDim>::template map<SuperCellSize > (threadIndex);

    /* "offset" from origin of the grid in unit of cells */
    const DataSpace<simDim> blockCell = block * SuperCellSize::toRT();

    __syncthreads();
    
    /* "particle box" : container/iterator where the particles live in 
     * and where one can get the frame in a super cell from */

    __shared__ IONFRAME *ionFrame;
    __shared__ ELECTRONFRAME *electronFrame;
    __shared__ bool isValid;
    __shared__ lcellId_t maxParticlesInFrame;

    __syncthreads(); /*wait that all shared memory is initialized*/

    /* find last frame in super cell 
     * define maxParticlesInFrame as the maximum frame size */
    if (linearThreadIdx == 0)
    {
        ionFrame = &(ionBox.getLastFrame(block, isValid));
        maxParticlesInFrame = PMacc::math::CT::volume<SuperCellSize>::type::value;
        /* @new - put in if other above does not work */
//        maxParticlesInFrame = PMacc::math::CT::volume<SuperCellSize>::type::value;
//        printf("mP1: %d ",maxParticlesInFrame);
    }

    __syncthreads();
    if (!isValid)
        return; //end kernel if we have no frames

    /* caching of E and B fields */
    PMACC_AUTO(cachedB, CachedBox::create < 0, typename BBox::ValueType > (BlockDescription_()));
    PMACC_AUTO(cachedE, CachedBox::create < 1, typename EBox::ValueType > (BlockDescription_()));

    __syncthreads();
    
    nvidia::functors::Assign assign;
    
    PMACC_AUTO(fieldBBlock, fieldB.shift(blockCell));
    ThreadCollective<BlockDescription_> collectiv(linearThreadIdx);
    collectiv(
              assign,
              cachedB,
              fieldBBlock
              );

    PMACC_AUTO(fieldEBlock, fieldE.shift(blockCell));
    collectiv(
              assign,
              cachedE,
              fieldEBlock
              );
    __syncthreads();

    /* Declare counter in shared memory that will later tell the current fill level or 
     * occupation of the newly created target electron frames. */
    __shared__ int newFrameFillLvl;
    
    __syncthreads(); /*wait that all shared memory is initialized*/
    
    /* Declare local variable oldFrameFillLvl for each thread */
    int oldFrameFillLvl;
    
    /* Initialize local (register) counter for each thread 
     * - describes how many new electrons should be created */
    int newElectrons = 0;
    
    /* Declare local electron ID
     * - describes at which position in the new frame the new electron is to be created */
    int electronId;
    
    /* Master initializes the frame fill level with 0 */
    if (linearThreadIdx == 0)
    {
        newFrameFillLvl = 0;        
        electronFrame = NULL;
    }
    __syncthreads();
    
    /* move over particle frames and call frameIonizer 
     * frames are worked on in backwards order to avoid asking if there is another frame
     * --> performance 
     * Because all frames are completely filled except the last and apart from that last frame 
     * one wants to make sure that all threads are working and every frame is worked on. */
    while (isValid)
    {
        /* casting uint8_t multiMask to boolean */
//        printf("mM: %d ",(*ionFrame)[linearThreadIdx][multiMask_]);
        bool isParticle = (*ionFrame)[linearThreadIdx][multiMask_];
        __syncthreads();
//        printf("iP: %d ",isParticle);

        /* < IONIZATION and change of charge states > 
         * if the threads contain particles, the frameIonizer can ionize them 
         * if they are non-particles their inner ionization counter remains at 0 */
        if (isParticle)
        {
        /* ionization based on ionization model */
//        printf("CALL! ");
        frameIonizer(*ionFrame, linearThreadIdx, cachedB, cachedE, newElectrons);
//        printf("nE: %d ", newElectrons);
        }
        __syncthreads();
        /* always true while-loop over all source frames */
        while (true)
        {
//            printf("INIT! ");
            /* < INIT >
             * - electronId is initialized as -1 (meaning: invalid)
             * - (local) oldFrameFillLvl set equal to (shared) newFrameFillLvl for each thread 
             * --> each thread remembers the old "counter"
             * - then sync */
            electronId = -1;
            oldFrameFillLvl = newFrameFillLvl;
            __syncthreads();
//            printf("ASK! ");
            /* < ASK >
             * - if a thread wants to create electrons in each cycle it can do that only once 
             * and before that it atomically adds to the shared counter and uses the current
             * value as electronId in the new frame 
             * - then sync */
            if (newElectrons > 0) // && isParticle between 1 and 27?
            {
                electronId = atomicAdd(&newFrameFillLvl, 1);
//                printf("%d ", electronId);
            }
            __syncthreads();
//            printf("EXIT? ");
            /* < EXIT? > 
             * - if the counter hasn't changed all threads break out of the loop */
            if (oldFrameFillLvl == newFrameFillLvl)
            {
//                printf("B! ");
                break;
            }
            __syncthreads();
//            printf("1st NEW FRAME! ");
            /* < FIRST NEW FRAME >
             * - if there is no frame, yet, the master will create a new target electron frame
             * and attach it to the back of the frame list 
             * - sync all threads again for them to know which frame to use */
            if (linearThreadIdx == 0)
            {
//                printf("%p ", electronFrame);
                if (electronFrame == NULL)
                {
                    electronFrame = &(electronBox.getEmptyFrame());
                    electronBox.setAsLastFrame(*electronFrame, block);
                }
//                printf("%p ",electronFrame);
            }
            __syncthreads();
//            printf("CREATE 1 ");
            /* < CREATE 1 >
             * - all electrons fitting into the current frame are created there 
             * - internal ionization counter is decremented by 1 
             * - sync */
            if ((0 <= electronId) && (electronId < maxParticlesInFrame))
            {
                /* each thread makes the attributes of its ion accessible */
                PMACC_AUTO(parentIon,((*ionFrame)[linearThreadIdx]));
                /* each thread initializes an electron if one should be created */
                PMACC_AUTO(targetElectronFull,((*electronFrame)[electronId]));
                
                /* create an electron in the new electron frame:
                 * - see particles/ionization/ionizationMethods.hpp */
                writeElectronIntoFrame(parentIon,targetElectronFull,*ionFrame,*electronFrame);
                
                newElectrons -= 1;
            }
            __syncthreads();
//            printf("2nd NEW FRAME ");
            /* < SECOND NEW FRAME > 
             * - if the shared counter is larger than the frame size a new electron frame is reserved
             * and attached to the back of the frame list
             * - then the shared counter is set back by one frame size
             * - sync so that every thread knows about the new frame */
            if (linearThreadIdx == 0)
            {
                if (newFrameFillLvl >= maxParticlesInFrame)
                {
                    electronFrame = &(electronBox.getEmptyFrame());
                    electronBox.setAsLastFrame(*electronFrame, block);
                    newFrameFillLvl -= maxParticlesInFrame;
                }
            }
            __syncthreads();
//            printf("CREATE 2 ");
            /* < CREATE 2 >
             * - if the EID is larger than the frame size 
             *      - the EID is set back by one frame size
             *      - the thread writes an electron to the new frame
             *      - the internal counter is decremented by 1 */
            if (electronId >= maxParticlesInFrame)
            {
                electronId -= maxParticlesInFrame;

                /* each thread makes the attributes of its ion accessible */
                PMACC_AUTO(parentIon,((*ionFrame)[linearThreadIdx]));
                /* each thread initializes an electron if one should be produced */
                PMACC_AUTO(targetElectronFull,((*electronFrame)[electronId]));
                
                /* create an electron in the new electron frame:
                 * - see particles/ionization/ionizationMethods.hpp */
                writeElectronIntoFrame(parentIon,targetElectronFull,*ionFrame,*electronFrame);
         
                newElectrons -= 1;
                
            }
            __syncthreads();
        }
        __syncthreads();

        if (linearThreadIdx == 0)
        {
            ionFrame = &(ionBox.getPreviousFrame(*ionFrame, isValid));
            maxParticlesInFrame = PMacc::math::CT::volume<SuperCellSize>::type::value;
//            printf("mP2: %d ",maxParticlesInFrame);
        }
        // isParticle = true;
        __syncthreads();
    }

}
    
#include "algorithms/Gamma.hpp"
#include "algorithms/Velocity.hpp"
//hard code
//namespace particleIonizerNone
//    {
//        template<class Velocity, class Gamma>
//        struct Ionize
//        {
//
//            template<typename EType, typename BType, typename PosType, typename MomType, typename MassType, typename ChargeType, typename ChargeStateType >
//                    __host__ DINLINE void operator()(
//                                                        const BType bField, /* at t=0 */
//                                                        const EType eField, /* at t=0 */
//                                                        PosType& pos, /* at t=0 */
//                                                        MomType& mom, /* at t=-1/2 */
//                                                        const MassType mass,
//                                                        const ChargeType charge,
//                                                        ChargeStateType& chState)
//            {
//                
//                /*Barrier Suppression Ionization for hydrogenlike helium 
//                 *charge >= 0 is needed because electrons and ions cannot be 
//                 *distinguished, yet.
//                 */
////                printf("cs: %d ",chState);
//                if (math::abs(eField)*UNIT_EFIELD >= 5.14e7 && chState < 2 && charge >= 0)
//                {
//                    chState = 1 + chState;
////                    printf("CS: %u ", chState);
//                }
//                
//                /*
//                 *int firstIndex = blockIdx.x * blockIdx.y * blockIdx.z * threadIdx.x * threadIdx.y * threadIdx.z;
//                 *if (firstIndex == 0)
//                 *{
//                 *    printf("Charge State: %u", chState);
//                 *}
//                 */
//                
//            }
//        };
//    } //namespace particleIonizerNone

//namespace particleIonizer = particleIonizerNone;

//#include "simulation_defines/unitless/ionizerConfig.unitless"

/* IONIZE (former UPDATE from Particles.tpp) */
template< typename T_ParticleDescription>
template< typename T_Elec>
void Particles<T_ParticleDescription>::ionize( T_Elec electrons, uint32_t )
{
    
    typedef typename HasFlag<FrameType,particleIonizer<> >::type hasIonizer;
    typedef typename GetFlagType<FrameType,particleIonizer<> >::type FoundIonizer;

    /* if no ionizer was defined we use IonizerNone as fallback */
//    typedef typename bmpl::if_<hasIonizer,FoundIonizer,particles::ionizer::None >::type SelectIonizer;
    typedef typename particles::ionizer::None::type SelectIonizer;
    /* that doesn't seem to work for the fallback particleIonizerNone */
//    typedef typename SelectIonizer::type ParticleIonize;
    typedef SelectIonizer ParticleIonize;

    typedef typename GetFlagType<FrameType,interpolation<> >::type::ThisType InterpolationScheme;

    /* margins around the supercell for the interpolation of the field on the cells */
    typedef typename GetMargin<InterpolationScheme>::LowerMargin LowerMargin;
    typedef typename GetMargin<InterpolationScheme>::UpperMargin UpperMargin;
    
    /* frame ionizer that moves over the frames with the particles and executes an operation on them */
    typedef IonizeParticlesPerFrame<ParticleIonize, MappingDesc::SuperCellSize,
        InterpolationScheme,
        fieldSolver::NumericalCellType > FrameIonizer;
    
    /* relevant area of a block */
    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        LowerMargin,
        UpperMargin
        > BlockArea;

    /* 3-dim vector : number of threads to be started in every dimension */
    dim3 block( MappingDesc::SuperCellSize::toRT().toDim3() );
    
    /* kernel call : instead of name<<<blocks, threads>>> (args, ...) 
       "blocks" will be calculated from "this->cellDescription" and "CORE + BORDER" 
       "threads" is calculated from the previously defined vector "block" */
//    printf("Call the Colonel!\n");
    __picKernelArea( kernelIonizeParticles<BlockArea>, this->cellDescription, CORE + BORDER )
        (block)
        ( this->getDeviceParticlesBox( ),
          electrons->getDeviceParticlesBox(),
          this->fieldE->getDeviceDataBox( ),
          this->fieldB->getDeviceDataBox( ),
          FrameIonizer( )
          );
    /* for testing purposes: enables more frequent output*/
    cudaDeviceSynchronize();

    /* fill the gaps in both species' particle frames to ensure that only 
     * the last frame is not completely filled but every other before is full */
    this->fillAllGaps( ); //"this" refers to ions here 
    electrons->fillAllGaps();
    
}

}