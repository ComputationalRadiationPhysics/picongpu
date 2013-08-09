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
 
/* 
 * File:   CoreFrame.hpp
 * Author: widera
 *
 * Created on 22. November 2010, 14:10
 */

#ifndef BORDERFRAME_HPP
#define	BORDERFRAME_HPP

#include "particles/frame_types.hpp"
#include "basisLib/vector/Vector.hpp"
#include "particles/memory/boxes/TileDataBox.hpp"

#include "particles/memory/frames/FrameCopy.hpp"

namespace PMacc
{


/**
 * A frame which holds border buffer data (in contrast to core data).
 *
 * @tparam SuperCellSize TVec which descripe size of a supercell
 * @tparam DIM dimension for this frame
 */
template<typename PositionType,class SuperCellSize_, unsigned DIM>
class BorderFrame
{
public:

    typedef PositionType PosType;

    typedef SuperCellSize_ SuperCellSize;

    enum
    {
        tileSize = SuperCellSize::elements, dim = DIM, FrameIdentifier = BORDER_FRAME
    };

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

    /**
     * Returns the Box with local cell idx.
     *
     * Idx are local ids from the supercell.
     *
     * @return TileDataBox with TILESIZE elements
     */
    HDINLINE VectorDataBox<lcellId_t> getCellIdx()
    {
        return VectorDataBox<lcellId_t > (this->cellIdx);
    }

    /**
     * Returns the Box with positions.
     *
     * Positions are positions in the cell relative to the top-left edge.
     *
     * @return TileDataBox with TILESIZE elements
     */
    HDINLINE VectorDataBox<PosType > getPosition()
    {
        return VectorDataBox<PosType > (this->position);
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


    /**\todo: return BitMasks */

private:

    PMACC_ALIGN(position[tileSize], PosType);
    PMACC_ALIGN(cellIdx[tileSize], lcellId_t);
};
}

#endif	/* BORDERFRAME_HPP */
