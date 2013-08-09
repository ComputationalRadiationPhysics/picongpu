/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, René Widera, Wolfgang Hoenig
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
 * File:   SubGrid.hpp
 * Author: wolfgang
 *
 * Created on 24. April 2010, 13:20
 */

#pragma once

#include "dimensions/DataSpace.hpp"
#include "mappings/simulation/GridController.hpp"
#include "dimensions/GridLayout.hpp"

namespace PMacc
{

template <unsigned DIM>
class SimulationBox
{
    typedef DataSpace<DIM> Size;
public:

    HDINLINE SimulationBox(const Size& localSize,
                          const Size& globalSize,
                          const Size& globalOffset) :
    localSize(localSize),
    globalSize(globalSize),
    globalOffset(globalOffset)
    {

    }

    HDINLINE SimulationBox()
    {

    }

    /** get size of global simulation box (in cells)
     *      
     */
    HDINLINE Size getGlobalSize() const
    {
        return globalSize;
    }

    /**
     * Get size of local simulation area.
     * 
     * @return size of local simulation area
     */
    HDINLINE Size getLocalSize() const
    {
        return localSize;
    }

    /**
     * Get distance from global origin to local origin (in cells)
     *
     * local null point=Top/Left in 2D, Top/Left/Front in 3D
     * 
     * @return offset to local origin of ordinates
     */
    HDINLINE Size getGlobalOffset() const
    {
        return globalOffset;
    }

    HDINLINE void setGlobalOffset(const Size& offset)
    {
        globalOffset = offset;
    }

private:
    Size globalSize;
    Size localSize;
    Size globalOffset;

};

template <unsigned DIM>
class SubGrid
{
public:

    typedef SubGrid<DIM> MyType;
    typedef DataSpace<DIM> Size;

    static MyType& getInstance()
    {
        static MyType instance;
        return instance;
    }

    void init(const Size localSize,
              const Size globalSize,
              const Size globalOffset)
    {
        SimulationBox<DIM> box(localSize, globalSize, globalOffset);
        simBox = box;
    }

    void setGlobalOffset(const Size& offset)
    {
        simBox.setGlobalOffset(offset);
    }

    const SimulationBox<DIM> getSimulationBox() const
    {
        return simBox;
    }

private:

    /**
     * Constructor
     */
    SubGrid()
    {

    }

    virtual ~SubGrid()
    {
    }

    /**
     * Constructor
     */
    SubGrid(const SubGrid& gc)
    {

    }


    SimulationBox<DIM> simBox;
};


} //namespace PMacc



