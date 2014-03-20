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
#include "Particles.hpp"
#include <cassert>


#include "particles/Particles.kernel"

#include "dataManagement/DataConnector.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"

#include "particles/memory/buffers/ParticlesBuffer.hpp"
#include "ParticlesInit.kernel"
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
#if(ENABLE_RADIATION == 1)
#include "plugins/radiation/particles/PushExtension.hpp"
#endif

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

/* UPDATE */

template< typename T_DataVector, typename T_MethodsVector>
void Particles<T_DataVector,T_MethodsVector>::update( uint32_t )
{
    /* particle pusher - to be replaced with ionization routine */
    typedef particlePusher::ParticlePusher ParticlePush;
    
    /* margins around the supercell for the interpolation of the field on the cells */
    typedef typename GetMargin<fieldSolver::FieldToParticleInterpolation>::LowerMargin LowerMargin;
    typedef typename GetMargin<fieldSolver::FieldToParticleInterpolation>::UpperMargin UpperMargin;
    
    /* frame solver that moves over the frames with the particles and executes an operation on them */
    typedef PushParticlePerFrame<ParticlePush, MappingDesc::SuperCellSize,
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
    __picKernelArea( kernelMoveAndMarkParticles<BlockArea>, this->cellDescription, CORE + BORDER )
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

/* MOVE AND MARK */

template<class BlockDescription_, class ParBox, class BBox, class EBox, class Mapping, class FrameSolver>
__global__ void kernelMoveAndMarkParticles(ParBox pb,
                                           EBox fieldE,
                                           BBox fieldB,
                                           FrameSolver frameSolver,
                                           Mapping mapper)
{
    /* definitions for domain variables, like indices of blocks and threads
     *  
     * conversion from block to linear frames */
    typedef typename BlockDescription_::SuperCellSize SuperCellSize;
    const DataSpace<simDim> block(mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx)));


    const DataSpace<simDim > threadIndex(threadIdx);
    const int linearThreadIdx = DataSpaceOperations<simDim>::template map<SuperCellSize > (threadIndex);


    const DataSpace<simDim> blockCell = block * SuperCellSize::getDataSpace();

    __syncthreads();


    __shared__ typename ParBox::FrameType *frame;
    __shared__ bool isValid;
    __shared__ int mustShift;
    __shared__ lcellId_t particlesInSuperCell;

    __syncthreads(); /*wait that all shared memory is initialized*/

    if (linearThreadIdx == 0)
    {
        mustShift = 0;
        frame = &(pb.getLastFrame(block, isValid));
        particlesInSuperCell = pb.getSuperCell(block).getSizeLastFrame();
    }

    __syncthreads();
    if (!isValid)
        return; //end kernel if we have no frames

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

    /* move over frames and call frame solver*/
    while (isValid)
    {
        if (linearThreadIdx < particlesInSuperCell)
        {
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

    /* PARTICLE PUSHER */
    namespace particlePusherFree
    {
        template<class Velocity, class Gamma>
        struct Push
        {

            template<typename EType, typename BType, typename PosType, typename MomType, typename MassType, typename ChargeType >
                    __host__ DINLINE void operator()(
                                                        const BType bField,
                                                        const EType eField,
                                                        PosType& pos,
                                                        MomType& mom,
                                                        const MassType mass,
                                                        const ChargeType charge)
            {

                Velocity velocity;
                const PosType vel = velocity(mom, mass);

                /* IMPORTANT: 
                 * use float_X(1.0)+X-float_X(1.0) because the rounding of float_X can create position from [-float_X(1.0),2.f],
                 * this breaks ower definition that after position change (if statements later) the position must [float_X(0.0),float_X(1.0))
                 * 1.e-9+float_X(1.0) = float_X(1.0) (this is not allowed!!!
                 * 
                 * If we don't use this fermi crash in this kernel in the time step n+1 in field interpolation
                 */
                pos.x() += float_X(1.0) + (vel.x() * DELTA_T / CELL_WIDTH);
                pos.y() += float_X(1.0) + (vel.y() * DELTA_T / CELL_HEIGHT);
                pos.z() += float_X(1.0) + (vel.z() * DELTA_T / CELL_DEPTH);

                pos.x() -= float_X(1.0);
                pos.y() -= float_X(1.0);
                pos.z() -= float_X(1.0);

            }
        };
    } //namespace

}