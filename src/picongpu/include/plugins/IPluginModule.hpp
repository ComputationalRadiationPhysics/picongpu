/**
 * Copyright 2013 Axel Huebl, René Widera
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
 


#ifndef IPluginModule_HPP
#define	IPluginModule_HPP

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"
#include "moduleSystem/ModuleConnector.hpp"
#include "moduleSystem/Module.hpp"


namespace picongpu
{
    using namespace PMacc;

    class IPluginModule : public Module
    {
    public:
        virtual void setMappingDescription(MappingDesc *cellDescription) = 0;

        virtual ~IPluginModule()
        {
        };
    };

}

#endif	/* IPluginModule_HPP */

