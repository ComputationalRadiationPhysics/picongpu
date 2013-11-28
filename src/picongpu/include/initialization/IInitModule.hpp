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
 


#ifndef IINITMODULE_HPP
#define	IINITMODULE_HPP

#include "types.h"
#include "moduleSystem/ModuleConnector.hpp"
#include "plugins/IPluginModule.hpp"


namespace picongpu
{
    using namespace PMacc;

    class IInitModule :  public IPluginModule
    {
    public:
        virtual void slide(uint32_t currentStep) = 0;
        virtual uint32_t init() = 0;

        virtual ~IInitModule()
        {
        }
        
    };
}

#endif	/* IINITMODULE_HPP */

