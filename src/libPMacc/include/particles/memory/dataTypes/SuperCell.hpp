/**
 * Copyright 2013-2017 Heiko Burau, Rene Widera
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

#include "pmacc_types.hpp"
#include "memory/Array.hpp"

namespace PMacc
{

template <class TYPE>
class SuperCell
{
public:
    /** number of cells in the super cell */
    typedef typename math::CT::volume<typename TYPE::SuperCellSize>::type CellsPerSupercell;
    /** array that holds the absolute in-frame position of the first particle for each cell */
    typedef memory::Array< uint32_t, CellsPerSupercell::value> CellEntryPointType;
    /** array that holds the number of particles in each cell for a certain species */
    typedef memory::Array< uint32_t, CellsPerSupercell::value> CellCountType;

    HDINLINE SuperCell() :
    firstFramePtr(NULL),
    lastFramePtr(NULL),
    mustShiftVal(false),
    sizeLastFrame(0)
    {
    }

    HDINLINE TYPE* FirstFramePtr()
    {
        return firstFramePtr;
    }

    HDINLINE TYPE* LastFramePtr()
    {
        return lastFramePtr;
    }

    HDINLINE const TYPE* FirstFramePtr() const
    {
        return firstFramePtr;
    }

    HDINLINE const TYPE* LastFramePtr() const
    {
        return lastFramePtr;
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
    PMACC_ALIGN(mustShiftVal, bool);
    PMACC_ALIGN(sizeLastFrame, lcellId_t);
public:
    PMACC_ALIGN(firstFramePtr, TYPE*);
    PMACC_ALIGN(lastFramePtr, TYPE*);
    PMACC_ALIGN(cellEntryPoint, CellEntryPointType);
    PMACC_ALIGN(cellCount, CellCountType);
};

} //end namespace
