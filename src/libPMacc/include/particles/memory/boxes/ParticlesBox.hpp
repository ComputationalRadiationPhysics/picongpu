/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
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

#include "particles/frame_types.hpp"
#include "dimensions/DataSpace.hpp"
#include "particles/memory/dataTypes/SuperCell.hpp"
#include "memory/boxes/PitchedBox.hpp"
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
    HINLINE ParticlesBox()
    {

    }

    HDINLINE ParticlesBox(const DataBox<PitchedBox<SuperCellType, DIM> > &superCells) :
    BaseType(superCells)
    {

    }

    /**
     * Returns an empty frame from data heap.
     *
     * @return an empty frame
     */
    HDINLINE FRAME &getEmptyFrame()
    {
        FrameType* tmp = new FrameType;

        if (tmp != NULL)
        {
            /* disable all particles since we can not assume that newly allocated memory contains zeros */
            for (int i = 0; i < (int) math::CT::volume<typename FrameType::SuperCellSize>::type::value; ++i)
                (*tmp)[i][multiMask_] = 0;
#if defined(__CUDA_ARCH__)
            /* takes care that changed values are visible to all threads inside this block*/
            __threadfence_block();
#endif
        }

        return *(FramePtr(tmp));
    }

    /**
     * Removes frame from heap data heap.
     *
     * @param frame FRAME to remove
     */
    HDINLINE void removeFrame(const FRAME &frame)
    {
        delete &frame;
    }

    /**
     * Returns the next frame in the linked list.
     *
     * @param frame the active FRAME
     * @return the next frame in the list
     */
    HDINLINE FRAME& getNextFrame(FRAME &frame, bool &isValid)
    {
        FramePtr tmp = frame.nextFrame;
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
        FramePtr tmp = frame.previousFrame;
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
        FramePtr tmp = FramePtr(getSuperCell(idx).LastFramePtr());
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
        FramePtr tmp = FramePtr(getSuperCell(idx).FirstFramePtr());
        isValid = tmp.isValid();
        return *tmp;

    }

    /**
     * Sets frame as the first frame of a supercell.
     *
     * @param frame frame to set as first frame
     * @param idx position of supercell
     */
    HDINLINE void setAsFirstFrame(FRAME &frameIn, const DataSpace<DIM> &idx)
    {
        FramePtr frame(&frameIn);
        FrameType** firstFrameNativPtr = &(getSuperCell(idx).firstFramePtr);

        frame->previousFrame = FramePtr();
        frame->nextFrame = FramePtr(*firstFrameNativPtr);
#if defined(__CUDA_ARCH__)
        /* - takes care that `next[index]` is visible to all threads on the gpu
         * - this is needed because later on in this method we change `previous`
         *   of an other frame, this must be done in order!
         */
        __threadfence();
#endif

#if !defined(__CUDA_ARCH__) // Host code path
        FramePtr oldFirstFramePtr(*firstFrameNativPtr);
        *firstFrameNativPtr = frame.ptr;
#else
        FramePtr oldFirstFramePtr((FrameType*) atomicExch((unsigned long long int*) firstFrameNativPtr, (unsigned long long int) frame.ptr));
#endif
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
    HDINLINE void setAsLastFrame(FRAME &frameIn, const DataSpace<DIM> &idx)
    {
        FramePtr frame(&frameIn);
        FrameType** lastFrameNativPtr = &(getSuperCell(idx).lastFramePtr);

        frame->nextFrame = FramePtr();
        frame->previousFrame = FramePtr(*lastFrameNativPtr);
#if defined(__CUDA_ARCH__)
        /* - takes care that `next[index]` is visible to all threads on the gpu
         * - this is needed because later on in this method we change `next`
         *   of an other frame, this must be done in order!
         */
        __threadfence();
#endif

#if !defined(__CUDA_ARCH__) // Host code path
        FramePtr oldLastFramePtr(*lastFrameNativPtr);
        *lastFrameNativPtr = frame.ptr;
#else
        FramePtr oldLastFramePtr((FrameType*) atomicExch((unsigned long long int*) lastFrameNativPtr, (unsigned long long int) frame.ptr));
#endif
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
    HDINLINE bool removeLastFrame(const DataSpace<DIM> &idx)
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
