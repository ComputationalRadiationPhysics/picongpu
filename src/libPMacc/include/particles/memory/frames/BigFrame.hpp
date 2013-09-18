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


#ifndef BIGFRAME_HPP
#define	BIGFRAME_HPP

#include "types.h"
#include "particles/frame_types.hpp"
#include "math/Vector.hpp"
#include "particles/memory/boxes/TileDataBox.hpp"

#include "FrameCopy.hpp"

namespace PMacc
{


/**
 * A large frame which can be used to merge data from several CoreFrames.
 * Primary aim is to improve IO operations using larger chunk sizes.
 *
 * @tparam SuperCellSize TVec which descripe size of a supercell
 * @tparam DIM dimension for this frame
 */
template<typename PositionType,class SuperCellSize_, unsigned DIM>
class BigFrame
{
    typedef PositionType PositionFloat;

public:
    typedef DataSpace<DIM> gcellId_t;

    typedef SuperCellSize_ SuperCellSize;

    enum
    {
        tileSize = SuperCellSize::elements, dim = DIM, FrameIdentifier = BIG_FRAME
    };

    /**
     * Returns the Box with global cell idx.
     *
     * Idx are global ids from the supercell.
     *
     * @return VectorDataBox with TILESIZE elements
     */
    HDINLINE VectorDataBox<gcellId_t> getGlobalCellIdx()
    {
        return VectorDataBox<gcellId_t > (this->cellIdx);
    }

    /**
     * Returns the Box with positions.
     *
     * Positions are positions in the cell relative to the top-left edge.
     *
     * @return VectorDataBox with TILESIZE elements
     */
    HDINLINE VectorDataBox<PositionFloat> getPosition()
    {
        return VectorDataBox<PositionFloat > (this->position);
    }

    /**
     * Returns the tile size.
     *
     * @return TILESIZE
     */
    HDINLINE lcellId_t getTileSize()
    {
        return tileSize;
    }

    /**
     * Copies all attributes of an element of the tile.
     *
     * @param myId id of the element in this frame
     * @param other another FRAME class to copy to
     * @param otherId id of the element in other frame
     */
    template<class FRAME>
    HDINLINE void copy(lcellId_t myId, FRAME &other, lcellId_t otherId)
    {
        FrameCopy::CopyFrame<FrameIdentifier, FRAME::FrameIdentifier> copyObj;
        copyObj.copy(*this, myId, other, otherId);
    }

private:
    PMACC_ALIGN(cellIdx[tileSize], gcellId_t);
    PMACC_ALIGN(position[tileSize], PositionFloat);

};

}

#endif	/* BIGFRAME_HPP */
