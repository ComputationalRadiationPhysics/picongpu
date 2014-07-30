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


#ifndef PARTICLESBOX_HPP
#define	PARTICLESBOX_HPP

#include "particles/frame_types.hpp"
#include "particles/memory/boxes/TileDataBox.hpp"
#include "particles/memory/boxes/HeapDataBox.hpp"
#include "dimensions/DataSpace.hpp"
#include "particles/memory/dataTypes/SuperCell.hpp"
#include "memory/boxes/PitchedBox.hpp"

namespace PMacc
{

/**
 * A DIM-dimensional Box holding frames with particle data.
 *
 * @tparam FRAME datatype for frames
 * @tparam DIM dimension of data (1-3)
 */
template<class FRAME, unsigned DIM>
class ParticlesBox
{
public:

    typedef FRAME FrameType;
    static const uint32_t Dim = DIM;

    HDINLINE ParticlesBox(const DataBox<PitchedBox<SuperCell<vint_t>, DIM> > &superCells,
                          const HeapDataBox<vint_t, FRAME>& data,
                          const VectorDataBox<vint_t>& nextFrames,
                          const VectorDataBox<vint_t>& prevFrames) :
    superCells(superCells), data(data), next(nextFrames), prev(prevFrames)
    {

    }

    /**
     * Returns an empty frame from data heap.
     *
     * @return an empty frame
     */
    HDINLINE FRAME &getEmptyFrame()
    {
        return data.pop();
    }

    /**
     * Removes frame from heap data heap.
     *
     * @param frame FRAME to remove
     */
    HDINLINE void removeFrame(FRAME &frame)
    {
        data.push(frame);
    }

    /**
     * Returns the next frame in the linked list.
     *
     * @param frame the active FRAME
     * @param isValid false if there is no next frame (returned frame is not unknown),
     * true otherwise (returned frame is the next frame)
     * @return the next frame in the list
     */
    HDINLINE FRAME& getNextFrame(FRAME &frame, bool &isValid)
    {
        vint_t myId = getFrameIdx(frame);
        vint_t nextId = next[myId];

        if (nextId == INV_IDX)
        {
            isValid = false;
            return frame;
        }
        else
        {
            isValid = true;
            return data[nextId];
        }
    }

    /**
     * Returns the previous frame in the linked list.
     *
     * @param frame the active FRAME
     * @param ret false if there is no previous frame (returned frame is not unknown),
     * true otherwise (returned frame is the previous frame)
     * @return the previous frame in the list
     */
    HDINLINE FRAME& getPreviousFrame(FRAME &frame, bool &ret)
    {
        vint_t myId = getFrameIdx(frame);
        vint_t prevId = prev[myId];

        if (prevId == INV_IDX)
        {
            ret = false;
            return frame;
        }
        else
        {
            ret = true;
            return data[prevId];
        }
    }

    /**
     * Returns the last frame of a supercell.
     *
     * @param idx position of supercell
     * @return the last FRAME of the linked list from supercell
     */
    HDINLINE FRAME& getLastFrame(const DataSpace<DIM> &idx, bool &isValid)
    {
        const vint_t frameId = superCells(idx).LastFrameIdx();
        isValid = (frameId != INV_IDX);
        if (isValid)
            return this->data[frameId];
        else
            return this->data[0];
    }

    /**
     * Returns the first frame of a supercell.
     *
     * @param idx position of supercell
     * @return the first FRAME of the linked list from supercell
     */
    HDINLINE FRAME& getFirstFrame(const DataSpace<DIM> &idx, bool &isValid)
    {
        const vint_t frameId = superCells(idx).FirstFrameIdx();
        isValid = (frameId != INV_IDX);
        if (isValid)
            return this->data[frameId];
        else
            return this->data[0];
    }

    /**
     * Sets frame as the first frame of a supercell.
     *
     * @param frame frame to set as first frame
     * @param idx position of supercell
     */
    HDINLINE void setAsFirstFrame(FRAME &frame, const DataSpace<DIM> &idx)
    {
        vint_t* firstFrameIdx = &(superCells(idx).FirstFrameIdx());
        vint_t index = getFrameIdx(frame);
        prev[index] = INV_IDX;
#if defined(__CUDA_ARCH__)
        /* - takes care that `prev[index]` is visible to all threads on the gpu
         * - this is needed because later on in this method we change `prev`
         *   of an other frame, this must be done in order!
         */
        __threadfence();
#endif
        next[index] = *firstFrameIdx;

        vint_t oldIndex;
#if !defined(__CUDA_ARCH__) // Host code path
        oldIndex = *firstFrameIdx;
        *firstFrameIdx = index;
#else
        oldIndex = atomicExch(firstFrameIdx, index);
#endif

        next[index] = oldIndex;
        if (oldIndex != INV_IDX)
        {
            prev[oldIndex] = index;
        }
        else
        {
            //we add the first frame in supercell
            superCells(idx).LastFrameIdx() = index;
        }
    }

    /**
     * Sets frame as the last frame of a supercell.
     *
     * @param frame frame to set as last frame
     * @param idx position of supercell
     */
    HDINLINE void setAsLastFrame(FRAME &frame, const DataSpace<DIM> &idx)
    {

        vint_t* lastFrameIdx = &(superCells(idx).LastFrameIdx());
        vint_t index = getFrameIdx(frame);
        next[index] = INV_IDX;
#if defined(__CUDA_ARCH__)
        /* - takes care that `next[index]` is visible to all threads on the gpu
         * - this is needed because later on in this method we change `next`
         *   of an other frame, this must be done in order!
         */
        __threadfence();
#endif
        prev[index] = *lastFrameIdx;

        vint_t oldIndex;
#if !defined(__CUDA_ARCH__) // Host code path
        oldIndex = *lastFrameIdx;
        *lastFrameIdx = index;
#else
        oldIndex = atomicExch(lastFrameIdx, index);
#endif

        prev[index] = oldIndex;
        if (oldIndex != INV_IDX)
        {
            next[oldIndex] = index;
        }
        else
        {
            //we add the first frame in supercell
            superCells(idx).FirstFrameIdx() = index;
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
        vint_t *lastFrameIdx = &(superCells(idx).LastFrameIdx());
        const vint_t last_id = *lastFrameIdx;

        const vint_t prev_id = prev[last_id];
        prev[last_id] = INV_IDX; //delete previous frame of the frame which we remove

        if (prev_id != INV_IDX)
        {
            //prev_id is Valid
            next[prev_id] = INV_IDX; //clear next of previous frame
            *lastFrameIdx = prev_id; //set new last particle
            removeFrame(data[last_id]);
            return true;
        }
        //remove last frame of supercell
        vint_t *firstFrameIdx = &(superCells(idx).FirstFrameIdx());
        *firstFrameIdx = INV_IDX;
        *lastFrameIdx = INV_IDX;
        removeFrame(data[last_id]);
        return false;
    }

    HDINLINE SuperCell<vint_t> &getSuperCell(DataSpace<DIM> idx)
    {
        return superCells(idx);
    }

private:

    HDINLINE vint_t getFrameIdx(const FRAME& frame) const
    {
        //const double x = (double) (sizeof (FRAME));
        //return (vint_t) floor(((double) ((size_t) (&frame) - (size_t)&(data[0])) / x + 0.00001));

        return ((size_t) (&frame) - (size_t) (&(data[0]))) / sizeof (FRAME);
    }

    PMACC_ALIGN8(superCells, DataBox<PitchedBox<SuperCell<vint_t>, DIM> >);
    PMACC_ALIGN8(data, HeapDataBox<vint_t, FRAME>);
    PMACC_ALIGN(next, VectorDataBox<vint_t>);
    PMACC_ALIGN(prev, VectorDataBox<vint_t>);

};

}
#endif	/* PARTICLESBOX_HPP */
