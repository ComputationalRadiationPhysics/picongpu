/* 
 * File:   ionization.hpp
 * Author: garten70
 *
 * Created on March 20, 2014, 9:21 AM
 */

#pragma once


/* includes from "update" */
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

#include "particles/memory/buffers/ParticlesBuffer.hpp"
#include "particles/ParticlesInit.kernel"
#include "mappings/simulation/GridController.hpp"

#include "simulationControl/MovingWindow.hpp"

#include <assert.h>
#include <limits>

#include "fields/numericalCellTypes/YeeCell.hpp"

/* includes from kernelMoveAndMarkParticles */
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

namespace picongpu
{
    
using namespace PMacc;

    

/* IONIZE PER FRAME (formerly PUSH PER FRAME from Particles.kernel) */

template<class IonizeAlgo, class TVec, class NumericalCellType>
struct IonizeParticlePerFrame
{

    template<class FrameType, class BoxB, class BoxE >
    DINLINE void operator()(FrameType& frame, int localIdx, BoxB& bBox, BoxE& eBox, int& mustShift, int& newElectrons)
    {

        typedef TVec Block;

        typedef typename BoxB::ValueType BType;
        typedef typename BoxE::ValueType EType;

        PMACC_AUTO(particle,frame[localIdx]);
        const float_X weighting = particle[weighting_];

        float3_X pos = particle[position_];
        const int particleCellIdx = particle[localCellIdx_];

        DataSpace<TVec::dim> localCell(DataSpaceOperations<TVec::dim>::template map<TVec > (particleCellIdx));

       
        EType eField = fieldSolver::FieldToParticleInterpolation()
            (eBox.shift(localCell).toCursor(), pos, NumericalCellType::getEFieldPosition());
        BType bField = fieldSolver::FieldToParticleInterpolation()
            (bBox.shift(localCell).toCursor(), pos,NumericalCellType::getBFieldPosition());

        float3_X mom = particle[momentum_];
        const float_X mass = frame.getMass(weighting);
        
        /*define charge state variable*/
        uint32_t chState = particle[chargeState_];
        
        IonizeAlgo ionizer;
        ionizer(
             bField, eField,
             pos,
             mom,
             mass,
             frame.getCharge(weighting, chState),
             chState
             );
        
        particle[momentum_] = mom;

        particle[position_] = pos;
        
        /* determine number of new electrons to be created */
        newElectrons = particle[chargeState_] - chState;
        /* retrieve new charge state */
        particle[chargeState_] = chState;

        /*calculate one dimensional cell index*/
        particle[localCellIdx_] = DataSpaceOperations<TVec::dim>::template map<TVec > (localCell);


        //particle[multiMask_] = direction;

    }
};

/* MARK PARTICLE (formerly MOVE AND MARK from Particles.kernel) */

/*template<class BlockDescription_, class ParBox, class BBox, class EBox, class Mapping, class FrameSolver>*/
template<class BlockDescription_, class IONFRAME, class ELECTRONFRAME, class BBox, class EBox, class Mapping, class FrameSolver>
__global__ void kernelIonizeParticles(ParticlesBox<IONFRAME, simDim> ionBox,
                                      ParticlesBox<ELECTRONFRAME, simDim> electronBox,
                                      EBox fieldE,
                                      BBox fieldB,
                                      FrameSolver frameSolver,
                                      Mapping mapper)
{
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
    const DataSpace<simDim> blockCell = block * SuperCellSize::getDataSpace();

    __syncthreads();

    typedef typename PIC_Ions::FrameType IONFRAME;
    typedef typename PIC_Electrons::FrameType ELECTRONFRAME;
    
    /* "particle box" : container/iterator where the particles live in 
     * and where one can get the frame in a super cell from */
    /*__shared__ typename ParBox::FrameType *frame;*/
    __shared__ IONFRAME *frame
    __shared__ ELECTRONFRAME *electronFrame
    __shared__ bool isValid;
    __shared__ int mustShift;
    __shared__ lcellId_t particlesInSuperCell;

    __syncthreads(); /*wait that all shared memory is initialized*/

    /* find last frame in super cell 
     * define particlesInSuperCell as the maximum frame size */
    if (linearThreadIdx == 0)
    {
        mustShift = 0;
        frame = &(ionBox.getLastFrame(block, isValid));
        particlesInSuperCell = ionBox.getSuperCell(block).getSizeLastFrame();
    }

    __syncthreads();
    if (!isValid)
        return; //end kernel if we have no frames

    /* caching of E and B fields */
    PMACC_AUTO(cachedB, CachedBox::create < 0, typename BBox::ValueType > (BlockDescription_()));
    PMACC_AUTO(fieldBBlock, fieldB.shift(blockCell));
    
    nvidia::functors::Assign assign;
    ThreadCollective<BlockDescription_> collectiv(linearThreadIdx);
    collectiv(
              assign,
              cachedB,
              fieldBBlock
              );
    PMACC_AUTO(cachedE, CachedBox::create < 1, typename EBox::ValueType > (BlockDescription_()));
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
    
    /* Declare local variable oldFrameFillLvl for each thread*/
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
    }
    
    /* move over frames and call frame solver 
     * frames are worked on in backwards order to avoid asking about if their is another frame
     * --> performance 
     * because all frames are completely filled except the last and apart from that last frame 
     * one wants to make sure that all threads are working and every frame is worked on */
    while (isValid)
    {
        /* always true while-loop over all source frames */
        while (true)
        {
            /* < IONIZATION and change of charge states > 
             * if the threads contain particles, the frameSolver can ionize them 
             * if they are non-particles their inner ionization counter remains at 0 */
            if (isParticle)
            {
                /* actual operation on the particles */
                frameSolver(*frame, linearThreadIdx, cachedB, cachedE, mustShift, newElectrons);
            }
            __syncthreads();
            /* < INIT >
             * - electronId is initialized as -1
             * - (local) oldFrameFillLvl set equal to (shared) newFrameFillLvl for each thread 
             * - then sync */
            electronId = -1;
            oldFrameFillLvl = newFrameFillLvl;
            __syncthreads();
            /* < ASK >
             * - if a thread wants to create electrons in each cycle it can do that only once 
             * and before that it atomically adds to the shared counter and uses the current
             * value as electronId in the new frame 
             * - then sync */
            if (newElectrons > 0) 
            {
                electronId = atomicAdd(&newFrameFillLvl, 1);
            }
            __syncthreads();
            /* < EXIT? > 
            * - if the counter hasn't changed all threads break out of the loop */
            if (oldFrameFillLvl == newFrameFillLvl)
            {
                break;
            }
            /* < FIRST NEW FRAME >
             * - if there is no frame, yet, the master will create a new target electron frame
             * and attach it to the back of the frame list 
             * - sync all again for them to know which frame to use */
            if (linearThreadIdx == 0)
            {
                if (oldFrameFillLvl != newFrameFillLvl)
                {
                    if (electronFrame == NULL)
                    {
                        electronFrame = &(electronBox.getEmptyFrame());
                        electronBox.setAsLastFrame(*electronFrame, block);
                    }
                }
            }
            __syncthreads();
            /* < CREATE 1 >
             * - all electrons fitting into the current frame are created there 
             * - internal ionization counter N is decremented by 1 
             * - sync */
            if (0 <= electronId < particlesInSuperCell)
            {
                /* < TASK > yet to be defined */
                /*writeElectronIntoFrame(*electronFrame,electronId);*/
                
                /* each thread makes the attributes of its ion accessible */
                PMACC_AUTO(parentIon,((*frame)[linearThreadIdx]));
                /* each thread initializes an electron if one should be produced */
                PMACC_AUTO(targetElectronFull,((*electronFrame)[electronId]));
                targetElectronFull[multiMask_] = 1;
                targetElectronFull[chargeState_] = 1;
                targetElectronFull[momentum_] = parentIon[momentum_]/parentIon.getMass(weighting)*targetElectronFull.getMass(weighting);

                /* each thread initializes a clone of the parent ion but leaving out
                 * some attributes:
                 * - multiMask: because it takes reportedly long to clone
                 * - chargeState: because electrons cannot have a charge state other than 1
                 * - momentum: because the electron would get a higher energy because of the ion mass */
                PMACC_AUTO(targetElectronClone, deselect<multiMask, chargeState, momentum>(targetElectronFull));
                assign(targetElectronClone, parentIon);
                
                newElectrons -= 1;
            }
            __syncthreads();
            /* < SECOND NEW FRAME > 
             * - if the shared counter is larger than the frame size a new electron frame is reserved
             * and attached to the back of the frame list
             * - then the shared counter is set back by one frame size
             * - sync so that every thread knows about the new frame */
            if (linearThreadIdx == 0)
            {
                if (newFrameFillLvl > particlesInSuperCell)
                {
                    electronFrame = &(electronBox.getEmptyFrame());
                    electronBox.setAsLastFrame(*electronFrame, block);
                    newFrameFillLvl -= particlesInSuperCell;
                }
            }
            __syncthreads();
            /* < CREATE 2 >
             * - if the EID is larger than the frame size 
             *      - the EID is set back by one frame size
             *      - the thread writes an electron to the new frame
             *      - the internal counter N is decremented by 1 */
            if (electronId >= particlesInSuperCell)
            {
                electronId -= particlesInSuperCell;
                /*writeElectronIntoFrame(*electronFrame,electronId);*/
                
                /* each thread makes the attributes of its ion accessible */
                PMACC_AUTO(parentIon,((*frame)[linearThreadIdx]));
                /* each thread initializes an electron if one should be produced */
                PMACC_AUTO(targetElectronFull,((*electronFrame)[electronId]));
                targetElectronFull[multiMask_] = 1;
                targetElectronFull[chargeState_] = 1;
                targetElectronFull[momentum_] = parentIon[momentum_]/parentIon.getMass(weighting)*targetElectronFull.getMass(weighting);

                /* each thread initializes a clone of the parent ion but leaving out
                 * some attributes:
                 * - multiMask: because it takes reportedly long to clone
                 * - chargeState: because electrons cannot have a charge state other than 1
                 * - momentum: because the electron would get a higher energy because of the ion mass */
                PMACC_AUTO(targetElectronClone, deselect<multiMask, chargeState, momentum>(targetElectronFull));
                assign(targetElectronClone, parentIon);
                
                newElectrons -= 1;
            }
        }

        if (linearThreadIdx == 0)
        {
            frame = &(ionBox.getPreviousFrame(*frame, isValid));
            particlesInSuperCell = SuperCellSize::elements;
        }
        // isParticle = true;
        __syncthreads();
        /* TODO <FillAllGaps> in electron frames */
    }

}
    
/* IONIZE (former UPDATE from Particles.tpp) */
#include "algorithms/Gamma.hpp"
#include "algorithms/Velocity.hpp"
//hard code
namespace particleIonizerNone
    {
        template<class Velocity, class Gamma>
        struct Ionize
        {

            template<typename EType, typename BType, typename PosType, typename MomType, typename MassType, typename ChargeType, typename ChargeStateType >
                    __host__ DINLINE void operator()(
                                                        const BType bField,
                                                        const EType eField,
                                                        PosType& pos,
                                                        MomType& mom,
                                                        const MassType mass,
                                                        const ChargeType charge,
                                                        ChargeStateType& chState)
            {
                
                /*Barrier Suppression Ionization for hydrogenlike helium 
                 *charge >= 0 is needed because electrons and ions cannot be 
                 *distinguished, yet.
                 */
                if (math::abs(eField)*UNIT_EFIELD >= 5.14e7 && chState < 2 && charge >= 0)
                {
                    chState = 1 + chState;
                }
                
                /*
                 *int firstIndex = blockIdx.x * blockIdx.y * blockIdx.z * threadIdx.x * threadIdx.y * threadIdx.z;
                 *if (firstIndex == 0)
                 *{
                 *    printf("Charge State: %u", chState);
                 *}
                 */
                
            }
        };
    
        typedef Ionize<Velocity, Gamma<> > ParticleIonizer;
    }

namespace particleIonizer = particleIonizerNone;

//end hard code

template< typename T_ParticleDescription ,typename T_Elec>
void Particles<T_ParticleDescription>::ionize( uint32_t, T_Elec electrons )
{
    /* particle ionization routine */
    
    typedef particleIonizer::ParticleIonizer ParticleIonize;
    
    /* margins around the supercell for the interpolation of the field on the cells */
    typedef typename GetMargin<fieldSolver::FieldToParticleInterpolation>::LowerMargin LowerMargin;
    typedef typename GetMargin<fieldSolver::FieldToParticleInterpolation>::UpperMargin UpperMargin;
    
    /* frame solver that moves over the frames with the particles and executes an operation on them */
    typedef IonizeParticlePerFrame<ParticleIonize, MappingDesc::SuperCellSize,
        fieldSolver::NumericalCellType > FrameSolver;
    
    /* relevant area of a block */
    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        typename toTVec<LowerMargin>::type,
        typename toTVec<UpperMargin>::type
        > BlockArea;

    /* 3-dim vector : number of threads to be started in every dimension */
    dim3 block( MappingDesc::SuperCellSize::getDataSpace( ) );

    /* kernel call : instead of name<<<blocks, threads>>> (args, ...) 
       "blocks" will be calculated from "this->cellDescription" and "CORE + BORDER" 
       "threads" is calculated from the previously defined vector "block" */
    //printf("Call the Colonel!\n");
    __picKernelArea( kernelIonizeParticles<BlockArea>, this->cellDescription, CORE + BORDER )
        (block)
        ( this->getDeviceParticlesBox( ),
          electrons->getDeviceParticlesBox(),
          this->fieldE->getDeviceDataBox( ),
          this->fieldB->getDeviceDataBox( ),
          FrameSolver( )
          );*/
    

    /* shift particles - WHERE TO? HOW?
     * fill the gaps in the simulation - WITH WHAT? */
    ParticlesBaseType::template shiftParticles < CORE + BORDER > ( );
    this->fillAllGaps( );
    
}

}