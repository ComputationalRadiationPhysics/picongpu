/**
 * Copyright 2013 Heiko Burau, René Widera
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


#ifndef POSITIONFILTER_HPP
#define	POSITIONFILTER_HPP

#include "types.h"
#include "particles/frame_types.hpp"
#include "particles/memory/frames/NullFrame.hpp"
#include "dimensions/DataSpaceOperations.hpp"

namespace PMacc
{


namespace privatePositionFilter
{

template<unsigned DIM, class Base = NullFrame>
class PositionFilter : public Base
{
protected:
    DataSpace<DIM> offset;
    DataSpace<DIM> max;
    DataSpace<DIM> superCellIdx;

public:

    HDINLINE PositionFilter()
    {
    }

    HDINLINE virtual ~PositionFilter()
    {
    }

    HDINLINE void setWindowPosition(DataSpace<DIM> offset, DataSpace<DIM> size)
    {
        this->offset = offset;
        this->max = offset + size;
    }

    HDINLINE void setSuperCellPosition(DataSpace<DIM> superCellIdx)
    {
        this->superCellIdx = superCellIdx;
    }

    HDINLINE DataSpace<DIM> getOffset()
    {
        return offset;
    }

};

} //namepsace privatePositionFilter

template<class Base = NullFrame>
class PositionFilter3D : public privatePositionFilter::PositionFilter<DIM3, Base>
{
public:

    template<class FRAME>
    HDINLINE bool operator()(FRAME & frame, lcellId_t id)
    {
        DataSpace<DIM3> localCellIdx3D = DataSpaceOperations<DIM3>::template map<
            typename FRAME::SuperCellSize
            > ((uint32_t) (frame[id][localCellIdx_]));
        DataSpace<DIM3> pos = this->superCellIdx + localCellIdx3D;
        return (this->offset.x() <= pos.x() && this->offset.y() <= pos.y() && this->offset.z() <= pos.z() &&
                this->max.x() > pos.x() && this->max.y() > pos.y() && this->max.z() > pos.z()) &&
            Base::operator() (frame, id);
    }
};

} //namepsace Frame

#endif	/* POSITIONFILTER_HPP */

