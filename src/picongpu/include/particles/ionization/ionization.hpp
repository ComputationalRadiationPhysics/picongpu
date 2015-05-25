/**
 * Copyright 2014 Marco Garten, Axel Huebl, Heiko Burau, Rene Widera,
 *                  Richard Pausch, Felix Schmitt
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


#include <iostream>

#include "simulation_defines.hpp"
#include "particles/Particles.hpp"
#include "mappings/kernel/AreaMapping.hpp"
#include "particles/ParticlesInit.kernel"
#include "mappings/simulation/GridController.hpp"
#include "simulationControl/MovingWindow.hpp"
#include "traits/Resolve.hpp"

#include "types.h"

#include "particles/ionization/ionizationMethods.hpp"

namespace picongpu
{
    
using namespace PMacc;

/** \struct IonizeParticlesPerFrame
 * \brief gathers fields for particle interpolation and gives them to the ionization model
 *
 * \tparam IonizeAlgo ionization algorithm
 * \tparam TVec dimensions of the target:
 *         e.g. size of the super cell
 * \tparam T_Field2ParticleInterpolation type of field to particle interpolation
 * \tparam NumericalCellType type of cell with respect to field discretization:
           e.g. Yee cell
 */
template<class IonizeAlgo, class TVec, class T_Field2ParticleInterpolation, class NumericalCellType>
struct IonizeParticlesPerFrame
{
    /** Functor implementation
     *
     * \tparam FrameType frame type of particles that get ionized
     * \tparam T_BBox type of B-field box
     * \tparam T_EBox type of E-field box
     *
     * \param ionFrame (here address of: ) frame of the to-be-ionized particles
     * \param localIdx local (linear) index in super cell / frame
     * \param bBox B-field box instance
     * \param ebox E-field box instance
     * \param newMacroElectrons (here address of: ) variable for each thread that stores the number
     *        of macro electrons to be created during the current time step
     */
    template<class FrameType, class T_BBox, class T_EBox >
    DINLINE void operator()(FrameType& ionFrame, int localIdx, T_BBox& bBox, T_EBox& eBox, unsigned int& newMacroElectrons)
    {

        typedef TVec Block;
        /** type for field to particle interpolation */
        typedef T_Field2ParticleInterpolation Field2ParticleInterpolation;

        typedef typename T_BBox::ValueType BType;
        typedef typename T_EBox::ValueType EType;

        PMACC_AUTO(particle,ionFrame[localIdx]);
        
        floatD_X pos = particle[position_];
        const int particleCellIdx = particle[localCellIdx_];

        DataSpace<TVec::dim> localCell(DataSpaceOperations<TVec::dim>::template map<TVec > (particleCellIdx));

       
        EType eField = Field2ParticleInterpolation()
            (eBox.shift(localCell).toCursor(), pos, NumericalCellType::getEFieldPosition());
        
        BType bField =Field2ParticleInterpolation()
            (bBox.shift(localCell).toCursor(), pos, NumericalCellType::getBFieldPosition());
        
        /* define number of bound macro electrons before ionization */
        float_X prevBoundElectrons = particle[boundElectrons_];
 
        /* this is the point where actual ionization takes place */
        IonizeAlgo ionizeAlgo;
        ionizeAlgo(
             bField, eField,
             particle
             );
        
        /* determine number of new macro electrons to be created */
        newMacroElectrons = prevBoundElectrons - particle[boundElectrons_];
        /*calculate one dimensional cell index*/

    }
}; // struct IonizeParticlesPerFrame


/** kernelIonizeParticles
 * \brief main kernel for ionization
 *
 * - maps the frame dimensions and gathers the particle boxes
 * - caches the E- and B-fields
 * - contains / calls the ionization algorithm
 * - calls the electron creation functors
 *
 * \tparam BlockDescription_ container for information about block / super cell dimensions
 * \tparam ParBoxIons container of the ions
 * \tparam ParBoxElectrons container of the electrons
 * \tparam BBox b-field box class
 * \tparam EBox e-field box class
 * \tparam Mapping class containing methods for acquiring info from the block
 * \tparam FrameIonizer \see IonizeParticlesPerFrame
 */
template<class BlockDescription_, class ParBoxIons, class ParBoxElectrons, class BBox, class EBox, class Mapping, class FrameIonizer>
__global__ void kernelIonizeParticles(ParBoxIons ionBox,
                                      ParBoxElectrons electronBox,
                                      EBox fieldE,
                                      BBox fieldB,
                                      FrameIonizer frameIonizer,
                                      Mapping mapper)
{
    
    /* "particle box" : container/iterator where the particles live in
     * and where one can get the frame in a super cell from */
    typedef typename ParBoxElectrons::FrameType ELECTRONFRAME;
    typedef typename ParBoxIons::FrameType IONFRAME;
    /* for not mixing assign up with the nVidia functor assign */
    namespace partOp = PMacc::particles::operations;
    
    /* definitions for domain variables, like indices of blocks and threads */
    typedef typename BlockDescription_::SuperCellSize SuperCellSize;
    /* "offset" 3D distance to origin in units of super cells */
    const DataSpace<simDim> block(mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx)));

    /* 3D vector from origin of the block to a cell in units of cells */
    const DataSpace<simDim > threadIndex(threadIdx);
    /* conversion from a 3D cell coordinate to a linear coordinate of the cell in its super cell */
    const int linearThreadIdx = DataSpaceOperations<simDim>::template map<SuperCellSize > (threadIndex);

    /* "offset" from origin of the grid in unit of cells */
    const DataSpace<simDim> blockCell = block * SuperCellSize::toRT();
    
    typedef typename particles::ionization::WriteElectronIntoFrame WriteElectronIntoFrame;

    __syncthreads();

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
    
    /* Declare counter in shared memory that will later tell the current fill level or
     * occupation of the newly created target electron frames. */
    __shared__ int newFrameFillLvl;
    
    __syncthreads(); /*wait that all shared memory is initialized*/
    
    /* Declare local variable oldFrameFillLvl for each thread */
    int oldFrameFillLvl;
    
    /* Initialize local (register) counter for each thread
     * - describes how many new macro electrons should be created */
    unsigned int newMacroElectrons = 0;
    
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
    
    /* move over source species frames and call frameIonizer
     * frames are worked on in backwards order to avoid asking if there is another frame
     * --> performance
     * Because all frames are completely filled except the last and apart from that last frame
     * one wants to make sure that all threads are working and every frame is worked on. */
    while (isValid)
    {
        /* casting uint8_t multiMask to boolean */
        const bool isParticle = (*ionFrame)[linearThreadIdx][multiMask_];
        __syncthreads();

        /* < IONIZATION and change of charge states >
         * if the threads contain particles, the frameIonizer can ionize them
         * if they are non-particles their inner ionization counter remains at 0 */
        if (isParticle)
            /* ionization based on ionization model - this actually increases charge states*/
            frameIonizer(*ionFrame, linearThreadIdx, cachedB, cachedE, newMacroElectrons);
       
        __syncthreads();
        /* always true while-loop over all particles inside source frame until each thread breaks out individually
         *
         * **Attention**: Speaking of 1st and 2nd frame only may seem odd.
         * The question might arise what happens if more electrons are created than would fit into two frames.
         * Well, multi-ionization during a time step is accounted for. The number of new electrons is
         * determined inside the outer loop over the valid frames while in the inner loop each thread can create only ONE
         * new macro electron. But the loop repeats until each thread has created all the electrons needed in the time step. */
        while (true)
        {
            /* < INIT >
             * - electronId is initialized as -1 (meaning: invalid)
             * - (local) oldFrameFillLvl set equal to (shared) newFrameFillLvl for each thread
             * --> each thread remembers the old "counter"
             * - then sync */
            electronId = -1;
            oldFrameFillLvl = newFrameFillLvl;
            __syncthreads();
            /* < CHECK & ADD >
             * - if a thread wants to create electrons in each cycle it can do that only once
             * and before that it atomically adds to the shared counter and uses the current
             * value as electronId in the new frame
             * - then sync */
            if (newMacroElectrons > 0)
                electronId = atomicAdd(&newFrameFillLvl, 1);
            
            __syncthreads();
            /* < EXIT? >
             * - if the counter hasn't changed all threads break out of the loop */
            if (oldFrameFillLvl == newFrameFillLvl)
                break;
            
            __syncthreads();
            /* < FIRST NEW FRAME >
             * - if there is no frame, yet, the master will create a new target electron frame
             * and attach it to the back of the frame list
             * - sync all threads again for them to know which frame to use */
            if (linearThreadIdx == 0)
            {
                if (electronFrame == NULL)
                {
                    electronFrame = &(electronBox.getEmptyFrame());
                    electronBox.setAsLastFrame(*electronFrame, block);
                }
            }
            __syncthreads();
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
                WriteElectronIntoFrame writeElectron;
                writeElectron(parentIon,targetElectronFull);
                
                newMacroElectrons -= 1;
            }
            __syncthreads();
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
                WriteElectronIntoFrame writeElectron;
                writeElectron(parentIon,targetElectronFull);
         
                newMacroElectrons -= 1;
            }
            __syncthreads();
        }
        __syncthreads();

        if (linearThreadIdx == 0)
        {
            ionFrame = &(ionBox.getPreviousFrame(*ionFrame, isValid));
            maxParticlesInFrame = PMacc::math::CT::volume<SuperCellSize>::type::value;
        }
        __syncthreads();
    }
} // void kernelIonizeParticles

/** ionize
 * \brief ionization function currently being a member function of the source species
 *
 * \tparam T_ParticleDescription container of particle source species information
 * \tparam T_Elec type of species to be created during ionization
 */
template< typename T_ParticleDescription >
template< typename T_Elec >
void Particles<T_ParticleDescription>::ionize( T_Elec electrons, uint32_t )
{
    
    /* get the alias for the used ionizer (ionization model) and specify the ionization algorithm */
    typedef typename GetFlagType<FrameType,ionizer<> >::type IonizerAlias;
    typedef typename PMacc::traits::Resolve<IonizerAlias>::type::IonizationAlgorithm IonizationAlgorithm;

    typedef typename PMacc::traits::Resolve<
        typename GetFlagType<FrameType,interpolation<> >::type
    >::type InterpolationScheme;

    /* margins around the supercell for the interpolation of the field on the cells */
    typedef typename GetMargin<InterpolationScheme>::LowerMargin LowerMargin;
    typedef typename GetMargin<InterpolationScheme>::UpperMargin UpperMargin;
    
    /* frame ionizer that moves over the frames with the particles and executes an operation on them */
    typedef IonizeParticlesPerFrame<IonizationAlgorithm, MappingDesc::SuperCellSize,
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
    __picKernelArea( kernelIonizeParticles<BlockArea>, this->cellDescription, CORE + BORDER )
        (block)
        ( this->getDeviceParticlesBox( ),
          electrons->getDeviceParticlesBox(),
          this->fieldE->getDeviceDataBox( ),
          this->fieldB->getDeviceDataBox( ),
          FrameIonizer( )
          );

    /* fill the gaps in the created species' particle frames to ensure that only
     * the last frame is not completely filled but every other before is full */
    electrons->fillAllGaps();
    
} // Particles::ionize

} // namespace picongpu
