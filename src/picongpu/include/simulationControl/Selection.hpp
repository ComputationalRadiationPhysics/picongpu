/**
 * Copyright 2014 Felix Schmitt
 *
 * This file is part of PIConGPU. 
 * 
 * PIConGPU is free software: you can redistribute it and/or modify 
 * it under the terms of the GNU General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * 
 * PIConGPU is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * along with PIConGPU.  
 * If not, see <http://www.gnu.org/licenses/>. 
 */

#pragma once

#include "types.h"

namespace picongpu
{
using namespace PMacc;

/**
 * Any DIM-dimensional selection of a simulation volume with a size and offset.
 * Can represent actual simulation data (domains) or overlays (windows).
 */
template <unsigned DIM>
class Selection
{
public:
    Selection()
    {
        
    }
    
    Selection(const Selection<DIM>& other) :
    size(other.size),
    offset(other.offset)
    {
        
    }
    
    Selection(DataSpace<DIM> size, DataSpace<DIM> offset) :
    size(size),
    offset(offset)
    {
        
    }
    
    DataSpace<DIM> size;

    DataSpace<DIM> offset;
};

} // namespace picongpu
