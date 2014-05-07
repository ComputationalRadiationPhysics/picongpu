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

#include "Selection.hpp"

namespace picongpu
{
using namespace PMacc;

/**
 * Groups local, global and total domain and window information.
 * 
 * For a detailed description of domains and windows, see the PIConGPU wiki page:
 * https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-domain-definitions
 */
template <unsigned DIM>
struct SelectionInformation
{
    /** total simulation volume, including active and inactive subvolumes */
    Selection<DIM> totalDomain;
    
    /** currently simulated volume over all GPUs, offset relative to totalDomain */
    Selection<DIM> globalDomain;

    /** currently simulated volume on this GPU, offset relative to globalDomain */
    Selection<DIM> localDomain;
    
    /** 
     * volume of the moving window (current simulation volume) across all GPUs
     * offset relative to globalDomain */
    Selection<DIM> globalMovingWindow;

    /** 
     * volume of the moving window (current simulation volume) on this GPU
     * offset relative to globalMovingWindow */
    Selection<DIM> localMovingWindow; 
    
    HINLINE const std::string toString(void) const
    {
        std::stringstream str;
        str << "[ totalDomain = " << totalDomain.toString() <<
                " globalDomain = " << globalDomain.toString() <<
                " localDomain = " << localDomain.toString() <<
                " globalMovingWindow = " << globalMovingWindow.toString() <<
                " localMovingWindow = " << localMovingWindow.toString() << " ]";
        
        return str.str();
    }
};

} // namespace picongpu

