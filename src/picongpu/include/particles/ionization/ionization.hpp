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
    DINLINE void operator()(FrameType& frame, int localIdx, BoxB& bBox, BoxE& eBox, int& mustShift)
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
        //uint32_t macroChState = frame.getChargeState(weighting, chState)
        
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
        /*retrieve new charge state*/
        particle[chargeState_] = chState;

        /*calculate one dimensional cell index*/
        particle[localCellIdx_] = DataSpaceOperations<TVec::dim>::template map<TVec > (localCell);


        //particle[multiMask_] = direction;

    }
};

/* MARK PARTICLE (formerly MOVE AND MARK from Particles.kernel) */

template<class BlockDescription_, class ParBox, class BBox, class EBox, class Mapping, class FrameSolver>
__global__ void kernelIonizeParticles(ParBox pb,
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

    /* "particle box" : container/iterator where the particles live in 
     * and where one can get the frame in a super cell from */
    __shared__ typename ParBox::FrameType *frame;
    __shared__ bool isValid;
    __shared__ int mustShift;
    __shared__ lcellId_t particlesInSuperCell;

    __syncthreads(); /*wait that all shared memory is initialized*/

    /* find last frame in super cell 
     * define particlesInSuperCell as the number of particles in that frame */
    if (linearThreadIdx == 0)
    {
        mustShift = 0;
        frame = &(pb.getLastFrame(block, isValid));
        particlesInSuperCell = pb.getSuperCell(block).getSizeLastFrame();
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

    /* move over frames and call frame solver 
     * frames are worked on in backwards order to avoid asking about if their is another frame
     * --> performance 
     * because all frames are completely filled except the last and apart from that last frame 
     * one wants to make sure that all threads are working and every frame is worked on */
    while (isValid)
    {
        if (linearThreadIdx < particlesInSuperCell)
        {
            /* actual operation on the particles */
            frameSolver(*frame, linearThreadIdx, cachedB, cachedE, mustShift);
        }
        __syncthreads();
        if (linearThreadIdx == 0)
        {
            frame = &(pb.getPreviousFrame(*frame, isValid));
            particlesInSuperCell = SuperCellSize::elements;
        }
        // isParticle = true;
        __syncthreads();
    }
    /* set the mustShift flag in SuperCell which is an optimization for shift particles and fillGaps*/
    if (linearThreadIdx == 0 && mustShift == 1)
    {
        pb.getSuperCell(mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx))).setMustShift(true);
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

template< typename T_ParticleDescription >
void Particles<T_ParticleDescription>::ionize( uint32_t )
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
          this->fieldE->getDeviceDataBox( ),
          this->fieldB->getDeviceDataBox( ),
          FrameSolver( )
          );

    /* shift particles - WHERE TO? HOW?
     * fill the gaps in the simulation - WITH WHAT? */
    ParticlesBaseType::template shiftParticles < CORE + BORDER > ( );
    this->fillAllGaps( );
    
}

}