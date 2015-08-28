/**
 * Copyright 2014-2015 Marco Garten, Axel Huebl, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Felix Schmitt
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
#include "nvidia/atomic.hpp"

#include "types.h"

#include "particles/ionization/ionizationMethods.hpp"

namespace picongpu
{
namespace particles
{
namespace ionization
{

using namespace PMacc;

/** kernelIonizeParticles
 * \brief main kernel for ionization
 *
 * - maps the frame dimensions and gathers the particle boxes
 * - contains / calls the ionization algorithm
 * - calls the electron creation functors
 *
 * \tparam ParBoxIons container of the ions
 * \tparam ParBoxElectrons container of the electrons
 * \tparam Mapping class containing methods for acquiring info from the block
 * \tparam FrameIonizer \see e.g. BSI_Impl in BSI_Impl.hpp
 *         instance of the ionization model functor
 */
template<class ParBoxIons, class ParBoxElectrons, class Mapping, class FrameIonizer>
__global__ void kernelIonizeParticles(ParBoxIons ionBox,
                                      ParBoxElectrons electronBox,
                                      FrameIonizer frameIonizer,
                                      Mapping mapper)
{

    /* "particle box" : container/iterator where the particles live in
     * and where one can get the frame in a super cell from
     */
    typedef typename ParBoxElectrons::FrameType ELECTRONFRAME;
    typedef typename ParBoxIons::FrameType IONFRAME;

    /* specify field to particle interpolation scheme */
    typedef typename PMacc::traits::Resolve<
        typename GetFlagType<IONFRAME,interpolation<> >::type
    >::type InterpolationScheme;

    /* margins around the supercell for the interpolation of the field on the cells */
    typedef typename GetMargin<InterpolationScheme>::LowerMargin LowerMargin;
    typedef typename GetMargin<InterpolationScheme>::UpperMargin UpperMargin;

    /* relevant area of a block */
    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        LowerMargin,
        UpperMargin
        > BlockDescription_;

    /* for not mixing operations::assign up with the nvidia functor assign */
    namespace partOp = PMacc::particles::operations;

    /* definitions for domain variables, like indices of blocks and threads */
    typedef typename BlockDescription_::SuperCellSize SuperCellSize;
    /* multi-dimensional offset vector from local domain origin on GPU in units of super cells */
    const DataSpace<simDim> block(mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx)));

    /* multi-dim vector from origin of the block to a cell in units of cells */
    const DataSpace<simDim > threadIndex(threadIdx);
    /* conversion from a multi-dim cell coordinate to a linear coordinate of the cell in its super cell */
    const int linearThreadIdx = DataSpaceOperations<simDim>::template map<SuperCellSize > (threadIndex);

    /* multi-dim offset from the origin of the local domain on GPU
     * to the origin of the block of the in unit of cells
     */
    const DataSpace<simDim> blockCell = block * SuperCellSize::toRT();

    /* subtract guarding cells to only have the simulation volume */
    const DataSpace<simDim> localCellIndex = (block * SuperCellSize::toRT() + threadIndex) - mapper.getGuardingSuperCells() * SuperCellSize::toRT();

    /* typedef for the functor that writes new macro electrons into electron frames during runtime */
    typedef typename particles::ionization::WriteElectronIntoFrame WriteElectronIntoFrame;

    __syncthreads();

    __shared__ IONFRAME *ionFrame;
    __shared__ ELECTRONFRAME *electronFrame;
    __shared__ bool isValid;
    __shared__ lcellId_t maxParticlesInFrame;

    __syncthreads(); /*wait that all shared memory is initialized*/

    /* find last frame in super cell
     * define maxParticlesInFrame as the maximum frame size
     */
    if (linearThreadIdx == 0)
    {
        ionFrame = &(ionBox.getLastFrame(block, isValid));
        maxParticlesInFrame = PMacc::math::CT::volume<SuperCellSize>::type::value;
    }

    __syncthreads();
    if (!isValid)
        return; //end kernel if we have no frames

    /* caching of E- and B- fields and initialization of random generator if needed */
    frameIonizer.init(blockCell, linearThreadIdx, localCellIndex);

    /* Declare counter in shared memory that will later tell the current fill level or
     * occupation of the newly created target electron frames.
     */
    __shared__ int newFrameFillLvl;

    __syncthreads(); /*wait until all shared memory is initialized*/

    /* Declare local variable oldFrameFillLvl for each thread */
    int oldFrameFillLvl;

    /* Initialize local (register) counter for each thread
     * - describes how many new macro electrons should be created
     */
    unsigned int newMacroElectrons = 0;

    /* Declare local electron ID
     * - describes at which position in the new frame the new electron is to be created
     */
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
     * one wants to make sure that all threads are working and every frame is worked on.
     */
    while (isValid)
    {
        /* casting uint8_t multiMask to boolean */
        const bool isParticle = (*ionFrame)[linearThreadIdx][multiMask_];
        __syncthreads();

        /* < IONIZATION and change of charge states >
         * if the threads contain particles, the frameIonizer can ionize them
         * if they are non-particles their inner ionization counter remains at 0
         */
        if (isParticle)
            /* ionization based on ionization model - this actually increases charge states*/
            frameIonizer(*ionFrame, linearThreadIdx, newMacroElectrons);

        __syncthreads();
        /* always true while-loop over all particles inside source frame until each thread breaks out individually
         *
         * **Attention**: Speaking of 1st and 2nd frame only may seem odd.
         * The question might arise what happens if more electrons are created than would fit into two frames.
         * Well, multi-ionization during a time step is accounted for. The number of new electrons is
         * determined inside the outer loop over the valid frames while in the inner loop each thread can create only ONE
         * new macro electron. But the loop repeats until each thread has created all the electrons needed in the time step.
         */
        while (true)
        {
            /* < INIT >
             * - electronId is initialized as -1 (meaning: invalid)
             * - (local) oldFrameFillLvl set equal to (shared) newFrameFillLvl for each thread
             * --> each thread remembers the old "counter"
             * - then sync
             */
            electronId = -1;
            oldFrameFillLvl = newFrameFillLvl;
            __syncthreads();
            /* < CHECK & ADD >
             * - if a thread wants to create electrons in each cycle it can do that only once
             * and before that it atomically adds to the shared counter and uses the current
             * value as electronId in the new frame
             * - then sync
             */
            if (newMacroElectrons > 0)
                electronId = nvidia::atomicAllInc(&newFrameFillLvl);

            __syncthreads();
            /* < EXIT? >
             * - if the counter hasn't changed all threads break out of the loop */
            if (oldFrameFillLvl == newFrameFillLvl)
                break;

            __syncthreads();
            /* < FIRST NEW FRAME >
             * - if there is no frame, yet, the master will create a new target electron frame
             * and attach it to the back of the frame list
             * - sync all threads again for them to know which frame to use
             */
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
             * - sync
             */
            if ((0 <= electronId) && (electronId < maxParticlesInFrame))
            {
                /* each thread makes the attributes of its ion accessible */
                PMACC_AUTO(parentIon,((*ionFrame)[linearThreadIdx]));
                /* each thread initializes an electron if one should be created */
                PMACC_AUTO(targetElectronFull,((*electronFrame)[electronId]));

                /* create an electron in the new electron frame:
                 * - see particles/ionization/ionizationMethods.hpp
                 */
                WriteElectronIntoFrame writeElectron;
                writeElectron(parentIon,targetElectronFull);

                newMacroElectrons -= 1;
            }
            __syncthreads();
            /* < SECOND NEW FRAME >
             * - if the shared counter is larger than the frame size a new electron frame is reserved
             * and attached to the back of the frame list
             * - then the shared counter is set back by one frame size
             * - sync so that every thread knows about the new frame
             */
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
             *      - the internal counter is decremented by 1
             */
            if (electronId >= maxParticlesInFrame)
            {
                electronId -= maxParticlesInFrame;

                /* each thread makes the attributes of its ion accessible */
                PMACC_AUTO(parentIon,((*ionFrame)[linearThreadIdx]));
                /* each thread initializes an electron if one should be produced */
                PMACC_AUTO(targetElectronFull,((*electronFrame)[electronId]));

                /* create an electron in the new electron frame:
                 * - see particles/ionization/ionizationMethods.hpp
                 */
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

} // namespace ionization
} // namespace particles
} // namespace picongpu
