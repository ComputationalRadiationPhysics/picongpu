/**
 * Copyright 2013-2015 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <mallocMC/mallocMC.hpp>
#include "particles/frame_types.hpp"
#include "dimensions/DataSpace.hpp"
#include "particles/memory/dataTypes/SuperCell.hpp"
#include "memory/boxes/PitchedBox.hpp"
#include "memory/boxes/DataBox.hpp"
#include "particles/memory/dataTypes/Pointer.hpp"

namespace PMacc
{

/**
 * A DIM-dimensional Box holding frames with particle data.
 *
 * @tparam FRAME datatype for frames
 * @tparam DIM dimension of data (1-3)
 */
template<class FRAME, unsigned DIM>
class ParticlesBox : protected DataBox<PitchedBox<SuperCell<FRAME>, DIM> >
{
private:
    PMACC_ALIGN(hostMemoryOffset,int64_t);
public:

    typedef FRAME FrameType;
    typedef Pointer<FrameType> FramePtr;
    typedef SuperCell<FrameType> SuperCellType;
    typedef DataBox<PitchedBox<SuperCell<FRAME>, DIM> > BaseType;

    static const uint32_t Dim = DIM;

    /** default constructor
     *
     * \warning after this call the object is in a invalid state and must be
     * initialized with an assignment of a valid ParticleBox
     */
    HDINLINE ParticlesBox() : hostMemoryOffset(0)
    {

    }

    HDINLINE ParticlesBox(const DataBox<PitchedBox<SuperCellType, DIM> > &superCells) :
    BaseType(superCells), hostMemoryOffset(0)
    {

    }

    HDINLINE ParticlesBox(const DataBox<PitchedBox<SuperCellType, DIM> > &superCells, int64_t memoryOffset) :
    BaseType(superCells), hostMemoryOffset(memoryOffset)
    {

    }

    /**
     * Returns an empty frame from data heap.
     *
     * @return an empty frame
     */
    DINLINE FRAME &getEmptyFrame()
    {
        FrameType* tmp = NULL;
        const int maxTries = 13; //magic number is not performance critical
        for (int numTries = 0; numTries < maxTries; ++numTries)
        {
            tmp = (FrameType*) mallocMC::malloc(sizeof (FrameType));
            if (tmp != NULL)
            {
                /* disable all particles since we can not assume that newly allocated memory contains zeros */
                for (int i = 0; i < (int) math::CT::volume<typename FrameType::SuperCellSize>::type::value; ++i)
                    (*tmp)[i][multiMask_] = 0;
                /* takes care that changed values are visible to all threads inside this block*/
                __threadfence_block();
                break;
            }
            else
            {
                printf("%s: mallocMC out of memory (try %i of %i)\n",
                       (numTries+1)==maxTries?"WARNING":"ERROR",
                       numTries+1,
                       maxTries);
            }
        }

        return *(FramePtr(tmp));
    }

    /**
     * Removes frame from heap data heap.
     *
     * @param frame FRAME to remove
     */
    DINLINE void removeFrame(FRAME &frame)
    {
        mallocMC::free((void*) &frame);
    }

    HDINLINE
    FrameType* mapPtr(FrameType* devPtr)
    {
#ifndef __CUDA_ARCH__
        int64_t useOffset=hostMemoryOffset*static_cast<int64_t>(devPtr!=0);
        return (FrameType*)(((char*)devPtr) - useOffset);
#else
        return devPtr;
#endif
    }

    /**
     * Returns the next frame in the linked list.
     *
     * @param frame the active FRAME
     * @return the next frame in the list
     */
    HDINLINE FRAME& getNextFrame(FRAME &frame, bool &isValid)
    {
        FramePtr tmp(mapPtr(frame.nextFrame.ptr));
        isValid = tmp.isValid();
        return *tmp;
    }

    /**
     * Returns the previous frame in the linked list.
     *
     * @param frame the active FRAME
     * @return the previous frame in the list
     */
    HDINLINE FRAME& getPreviousFrame(FRAME &frame, bool &isValid)
    {
        FramePtr tmp(mapPtr(frame.previousFrame.ptr));
        isValid = tmp.isValid();
        return *tmp;
    }

    /**
     * Returns the last frame of a supercell.
     *
     * @param idx position of supercell
     * @return the last FRAME of the linked list from supercell
     */
    HDINLINE FRAME& getLastFrame(const DataSpace<DIM> &idx, bool &isValid)
    {
        FramePtr tmp = FramePtr(mapPtr(getSuperCell(idx).LastFramePtr()));
        isValid = tmp.isValid();
        return *tmp;
    }

    /**
     * Returns the first frame of a supercell.
     *
     * @param idx position of supercell
     * @return the first FRAME of the linked list from supercell
     */
    HDINLINE FRAME& getFirstFrame(const DataSpace<DIM> &idx, bool &isValid)
    {
        FramePtr tmp = FramePtr(mapPtr(getSuperCell(idx).FirstFramePtr()));
        isValid = tmp.isValid();
        return *tmp;

    }

    /**
     * Sets frame as the first frame of a supercell.
     *
     * @param frame frame to set as first frame
     * @param idx position of supercell
     */
    DINLINE void setAsFirstFrame(FRAME &frameIn, const DataSpace<DIM> &idx)
    {
        FramePtr frame(&frameIn);
        FrameType** firstFrameNativPtr = &(getSuperCell(idx).firstFramePtr);

        frame->previousFrame = FramePtr();
        frame->nextFrame = FramePtr(*firstFrameNativPtr);

        /* - takes care that `next[index]` is visible to all threads on the gpu
         * - this is needed because later on in this method we change `previous`
         *   of an other frame, this must be done in order!
         */
        __threadfence();

        FramePtr oldFirstFramePtr((FrameType*) atomicExch((unsigned long long int*) firstFrameNativPtr, (unsigned long long int) frame.ptr));

        frame->nextFrame = oldFirstFramePtr;
        if (oldFirstFramePtr.isValid())
        {
            oldFirstFramePtr->previousFrame = frame;
        }
        else
        {
            //we add the first frame in supercell
            getSuperCell(idx).lastFramePtr = frame.ptr;
        }
    }

    /**
     * Sets frame as the last frame of a supercell.
     *
     * @param frame frame to set as last frame
     * @param idx position of supercell
     */
    DINLINE void setAsLastFrame(FRAME &frameIn, const DataSpace<DIM> &idx)
    {
        FramePtr frame(&frameIn);
        FrameType** lastFrameNativPtr = &(getSuperCell(idx).lastFramePtr);

        frame->nextFrame = FramePtr();
        frame->previousFrame = FramePtr(*lastFrameNativPtr);
        /* - takes care that `next[index]` is visible to all threads on the gpu
         * - this is needed because later on in this method we change `next`
         *   of an other frame, this must be done in order!
         */
        __threadfence();

        FramePtr oldLastFramePtr((FrameType*) atomicExch((unsigned long long int*) lastFrameNativPtr, (unsigned long long int) frame.ptr));

        frame->previousFrame = oldLastFramePtr;
        if (oldLastFramePtr.isValid())
        {
            oldLastFramePtr->nextFrame = frame;
        }
        else
        {
            //we add the first frame in supercell
            getSuperCell(idx).firstFramePtr = frame.ptr;
        }
    }

    /**
     * Removes the last frame of a supercell.
     * This call is not threadsave, only one thread from a supercell may call this function.
     * @param idx position of supercell
     * @return true if more frames in list, else false
     */
    DINLINE bool removeLastFrame(const DataSpace<DIM> &idx)
    {
        //!\todo this is not thread save
        FrameType** lastFrameNativPtr = &(getSuperCell(idx).lastFramePtr);

        FramePtr last(*lastFrameNativPtr);
        if (last.isValid())
        {
            FramePtr prev(last->previousFrame);

            if (prev.isValid())
            {
                prev->nextFrame = FramePtr(); //set to invalid frame
                *lastFrameNativPtr = prev.ptr; //set new last frame
                removeFrame(*last);
                return true;
            }
            //remove last frame of supercell
            getSuperCell(idx).firstFramePtr = NULL;
            getSuperCell(idx).lastFramePtr = NULL;

            removeFrame(*last);
        }
        return false;
    }

    HDINLINE SuperCellType& getSuperCell(DataSpace<DIM> idx)
    {
        return BaseType::operator()(idx);
    }

};

}
