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
 
/* 
 * File:   SuperCellBuffer.hpp
 * Author: widera
 *
 * Created on 7. Februar 2011, 07:57
 */

#ifndef SUPERCELL_HPP
#define	SUPERCELL_HPP

#include "types.h"


namespace PMacc
{



template <class TYPE>
class SuperCell
{
public:

    HDINLINE SuperCell() :
    firstFrameIdx(INV_IDX),
    lastFrameIdx(INV_IDX),
    mustShiftVal(false),
    sizeLastFrame(0)
    {
    }

    virtual ~SuperCell()
    {

    }

    HDINLINE TYPE& FirstFrameIdx()
    {
        return firstFrameIdx;
    }

    HDINLINE TYPE& LastFrameIdx()
    {
        return lastFrameIdx;
    }

    HDINLINE TYPE FirstFrameIdx() const
    {
        return firstFrameIdx;
    }

    HDINLINE TYPE LastFrameIdx() const
    {
        return lastFrameIdx;
    }

    HDINLINE bool mustShift()
    {
        return mustShiftVal;
    }

    HDINLINE void setMustShift(bool value)
    {
        mustShiftVal = value;
    }

    HDINLINE lcellId_t getSizeLastFrame()
    {
        return sizeLastFrame;
    }

    HDINLINE void setSizeLastFrame(lcellId_t size)
    {
        sizeLastFrame = size;
    }

private:


    PMACC_ALIGN(firstFrameIdx, TYPE);
    PMACC_ALIGN(lastFrameIdx, TYPE);
    PMACC_ALIGN(mustShiftVal, bool);
    PMACC_ALIGN(sizeLastFrame, lcellId_t);
};

} //end namespace

#endif	/* SUPERCELL_HPP */

