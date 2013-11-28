/**
 * Copyright 2013 Rene Widera
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
 

#ifndef _SIMULATION_DEFINES_HPP
#define _SIMULATION_DEFINES_HPP

#include <stdint.h>
#include "types.h"
#include <simulation_types.hpp>


namespace picongpu
{
    using namespace PMacc;
}


//##### load param
#include <simulation_defines/_defaultParam.loader>
#include <simulation_defines/extensionParam.loader>

//load starter after all user extension
#include <simulation_defines/param/starter.param>

#include <simulation_defines/param/componentsConfig.param>
#include <simulation_classTypes.hpp>

// ##### load unitless
#include <simulation_defines/_defaultUnitless.loader>
#include <simulation_defines/extensionUnitless.loader>
//load starter after user extensions and all params are loaded
#include <simulation_defines/unitless/starter.unitless>

#endif  /* _SIMULATION_DEFINES_HPP */
