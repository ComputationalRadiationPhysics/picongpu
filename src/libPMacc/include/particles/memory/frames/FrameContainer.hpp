/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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

#ifndef FRAMECONTAINER_HPP
#define	FRAMECONTAINER_HPP

#include <cassert>


#include "debug/DebugDataSpace.hpp"

#include "types.h"
#include "particles/frame_types.hpp"
#include "particles/memory/boxes/ParticlesBox.hpp"
#include "eventSystem/EventSystem.hpp"
#include "particles/memory/dataTypes/SuperCell.hpp"
#include "particles/memory/frames/BigFrame.hpp"
#include "particles/memory/frames/CoreFrame.hpp"
#include "particles/memory/buffers/ParticlesBuffer.hpp"

#include <boost/mpl/vector.hpp>
#include "particles/boostExtension/InheritGenerators.hpp"
#include "particles/boostExtension/JoinVectors.hpp"

namespace PMacc
{



/**
 * Helper class for FrameContainer used for computing the
 * next super cell position.
 *
 * @tparam DIM dimension for this class
 */
template<unsigned DIM>
class NextSuperCellPos
{
public:
    /**
     * Increases superCellPos to point to the next super cell.
     * Only positions in between [guardSuperCells, superCellsCount - guardSuperCells]
     * are returned.
     * 
     * @param guardSuperCells size of the guard (one side) in supercells in each dimension
     * @param superCellsCount total number of supercells, including guarding super cells, in each dimension
     * @param superCellPos the current super cell. must be in BORDER+CORE, not in GUARD region
     * @return true if another super cell position was found, false otherwise
     */
    static bool getNextSuperCellPos(DataSpace<DIM> guardSuperCells,
                                    DataSpace<DIM> superCellsCount, DataSpace<DIM>& superCellPos);
};

template<>
class NextSuperCellPos<DIM2>
{
public:

    static bool getNextSuperCellPos(DataSpace<DIM2> guardSuperCells,
                                    DataSpace<DIM2> superCellsCount, DataSpace<DIM2>& superCellPos)
    {
        // std::cout<<superCellsCount.x()<<" "<<superCellsCount.y()<<std::endl;
        if (superCellPos.x() + 1 < superCellsCount.x() - guardSuperCells.x())
        {
            superCellPos.x()++;
        }
        else
        {
            superCellPos.y()++;
            superCellPos.x() = guardSuperCells.x();
        }

        return (superCellPos.y() < superCellsCount.y() - guardSuperCells.y());
    }
};

template<>
class NextSuperCellPos<DIM3>
{
public:

    static bool getNextSuperCellPos(DataSpace<DIM3> guardSuperCells,
                                    DataSpace<DIM3> superCellsCount, DataSpace<DIM3>& superCellPos)
    {
        // std::cout<<superCellsCount.x()<<" "<<superCellsCount.y()<<std::endl;
        if (superCellPos.x() + 1 < superCellsCount.x() - guardSuperCells.x())
        {
            superCellPos.x()++;
        }
        else
        {
            superCellPos.x() = guardSuperCells.x();

            if (superCellPos.y() + 1 < superCellsCount.y() - guardSuperCells.y())
            {
                superCellPos.y()++;
            }
            else
            {
                superCellPos.y() = guardSuperCells.y();
                superCellPos.z()++;
            }

        }

        return (superCellPos.z() < superCellsCount.z() - guardSuperCells.z());
    }
};

/**
 * Helper class for FrameContainer used for converting local
 * positions to global positions.
 *
 * @param DIM dimension (2-3) for this class
 */
template<unsigned tileWidth, unsigned tileHeight, unsigned DIM>
class ConvertPosition
{
public:
    /**
     * Converts the id of cell frameCellId in the
     * super cell with superCellOffset into its position.
     * 
     * @param superCellOffset offset in cells of the super cell frameCellId is in
     * @param frameCellId id of the cell in the super cell
     * @return position of the cell with id frameCellId
     */
    static DataSpace<DIM> convertPosition(
                                          DataSpace<DIM>& superCellOffset,
                                          lcellId_t frameCellId);
};

template<unsigned tileWidth, unsigned tileHeight>
class ConvertPosition<tileWidth, tileHeight, DIM2>
{
public:

    static DataSpace<DIM2> convertPosition(
                                           DataSpace<DIM2>& superCellOffset,
                                           lcellId_t frameCellId)
    {
        DataSpace<DIM2> lCellIdx;
        lCellIdx.x() = frameCellId % tileWidth;
        lCellIdx.y() = frameCellId / tileWidth;
        return (superCellOffset + lCellIdx);
    }
};

template<unsigned tileWidth, unsigned tileHeight>
class ConvertPosition<tileWidth, tileHeight, DIM3>
{
public:

    static DataSpace<DIM3> convertPosition(

                                           DataSpace<DIM3>& superCellOffset,
                                           lcellId_t frameCellId)
    {
        DataSpace<DIM3> lCellIdx;
        size_t slice_size = tileWidth * tileHeight;
        lCellIdx.z() = frameCellId / (slice_size);
        frameCellId -= (slice_size * lCellIdx.z());

        lCellIdx.x() = frameCellId % tileWidth;
        lCellIdx.y() = frameCellId / tileWidth;
        return (superCellOffset + lCellIdx);
    }
};

/**
 * Iterator-like class which can be used to assemble CoreFrames to larger 
 * BigFrames and iterate over them.
 *
 * @param ParBuffer a ParticlesBuffer holding frame data
 * @param TILESIZE_BIG tilesize of a BigFrame, must be >= TILESIZE_SMALL
 * @param SuperCellVector size of a supercell in any direction
 */
template<class ParBuffer, class BigSuperCellSize, class SuperCellSize, class Filter>
class FrameContainer;

template<template <typename, typename, class, unsigned> class ParBuffer,
class BigSuperCellSize, class SuperCellSize, typename PositionType, typename UserTypeList,
unsigned DIM, class Filter>
class FrameContainer<ParBuffer<PositionType, UserTypeList, SuperCellSize, DIM>, BigSuperCellSize, SuperCellSize, Filter>
{
public:

    typedef ParBuffer<PositionType, UserTypeList, SuperCellSize, DIM> MyParBuffer;

    typedef ParticlesBox< typename MyParBuffer::ParticleType, DIM> MyParticlesBox;
    
    typedef typename JoinVectors<
        UserTypeList,
        bmpl::vector<BigFrame<PositionType,BigSuperCellSize, DIM> > 
    >::type full_listContainer;
    
    typedef typename LinearInherit<full_listContainer>::type ParticleType;

    enum
    {
        tileSizeBig = BigSuperCellSize::elements, tileSizeSmall = SuperCellSize::elements
    };

    /**
     * Constructor.
     *
     * @param particlesBuffer ParBuffer holding the frame data to merge
     * @param guardSuperCells number of super cells of border in each dimension
     */
    FrameContainer(MyParBuffer& particlesBuffer, DataSpace<DIM> guardSuperCells, Filter filter) :
    particlesBuffer(particlesBuffer),
    parBox(particlesBuffer.getHostParticleBox()),
    currentFrame(NULL),
    bigFrameElemCount(0),
    superCellPos(guardSuperCells),
    guardSuperCells(guardSuperCells),
    filter(filter)
    {
        clear();
    }

    void clear()
    {
        assert(tileSizeBig >= tileSizeSmall);
        totalElements = 0;
        bigFrameElemCount = 0;
        superCellsCount = particlesBuffer.getSuperCellsCount();
        superCellPos = guardSuperCells;
        currentFrame = NULL;
    }

    /**
     * Destructor.
     */
    virtual ~FrameContainer()
    {
        //std::cout << "total elements written: " << totalElements << std::endl;
    }

    /**
     * Returns the number of valid elements in the big frame.
     *
     * @return number of valid elements
     */
    size_t getElemCount()
    {
        return bigFrameElemCount;
    }

    /**
     * Iterates over all small CoreFrames of the ParticlesBuffer.
     *
     * BigFrames are filled with valid data from (tileSizeBig / tileSizeSmall) CoreFrames.
     * Therefor, returned big frames may not be filled completely but with getElemCount() elements only.
     * Relative position data is converted to global position information.
     * A filled BigFrame is returned.
     *
     * @param hasNextBigFrame returns if another big frame can be constructed (there are remaining small frames)
     * @return a filled BigFrame
     */
    ParticleType& getNextBigFrame(bool &hasNextBigFrame)
    {
        bigFrameElemCount = 0;

        hasNextBigFrame = false;

        bool frameValid = true;
        bool elementsFree = true;

        // if currentFrame is NULL, get the first frame from the first superCellPos
        if (currentFrame == NULL)
        {
            hasNextBigFrame = true;
            currentFrame = &(parBox.getFirstFrame(superCellPos, frameValid));
            while (frameValid == false && hasNextBigFrame == true)
            {
                // get the next valid super cell position
                // if there is none, no more BigFrames can be constructed
                if (NextSuperCellPos<DIM>::getNextSuperCellPos(
                                                               guardSuperCells,
                                                               superCellsCount,
                                                               superCellPos))
                {
                    currentFrame = &(parBox.getFirstFrame(superCellPos, frameValid));
                }
                else
                {
                    hasNextBigFrame = false;
                }
            }
        }

        while (frameValid && elementsFree)
        {
            hasNextBigFrame = true;
            typename CoreFrame<PositionType, SuperCellSize, DIM>::MultiMask multiMask = currentFrame->getMultiMask();

            // superCellOffset is the logical start of the supercell at superCellPos in cells, ignoring guards
            DataSpace<DIM> superCellOffset = (superCellPos - guardSuperCells) * particlesBuffer.getSuperCellSize();

            this->filter.setSuperCellPosition(superCellOffset);

            // iterate over all elements in the current frame
            for (uint32_t i = 0; i < tileSizeSmall; i++)
            {
                if (multiMask[i] && filter(*currentFrame, i))
                {
                    bigFrame.copy(bigFrameElemCount, *currentFrame, i);

                    // gCellPos is the GPU-global position of the cell 
                    DataSpace<DIM> gCellPos = ConvertPosition<SuperCellSize::x, SuperCellSize::y, DIM>::convertPosition(
                                                                                                                        superCellOffset, currentFrame->getCellIdx()[i]);

                    // set the respective values to gCellPos[offset].
                    bigFrame.getGlobalCellIdx()[bigFrameElemCount] = gCellPos;

                    bigFrameElemCount++;
                }
            }
            // compute free elements in current BigFrame
            elementsFree = (tileSizeBig - bigFrameElemCount) >= tileSizeSmall;
            // get next frame
            currentFrame = &(parBox.getNextFrame(*currentFrame, frameValid));

            while (frameValid == false && hasNextBigFrame == true)
            {
                // get the next valid super cell position
                // if there is none, no more BigFrames can be constructed
                if (NextSuperCellPos<DIM>::getNextSuperCellPos(
                                                               guardSuperCells,
                                                               superCellsCount,
                                                               superCellPos))
                {
                    currentFrame = &(parBox.getFirstFrame(superCellPos, frameValid));
                }
                else
                {
                    hasNextBigFrame = false;
                }
            }
        }

        totalElements += bigFrameElemCount;
        return bigFrame;
    }

    Filter& getFilter()
    {
        return filter;
    }

private:
    ParticleType bigFrame;
    MyParBuffer& particlesBuffer;
    MyParticlesBox parBox;

    size_t bigFrameElemCount;
    size_t totalElements;

    DataSpace<DIM> superCellPos;
    DataSpace<DIM> superCellsCount;
    typename MyParBuffer::ParticleType *currentFrame;
    DataSpace<DIM> guardSuperCells;

    Filter filter;
};



}

#endif	/* FRAMECONTAINER_HPP */

