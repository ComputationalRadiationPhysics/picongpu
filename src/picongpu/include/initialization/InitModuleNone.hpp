/**
 * Copyright 2013 René Widera
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
 


#ifndef INITMODULENONE_HPP
#define	INITMODULENONE_HPP

#include "initialization/IInitModule.hpp"



namespace picongpu
{
    using namespace PMacc;

    class InitModuleNone : public IInitModule
    {
    public:

        virtual void slide(uint32_t currentStep)
        {
        }

        virtual uint32_t init()
        {
            return 0;
        }

        virtual ~InitModuleNone()
        {
        }

        virtual void moduleRegisterHelp(po::options_description& desc)
        {
        }

        virtual std::string moduleGetName() const
        {
            return "InitModuleNone";
        }

        virtual void setMappingDescription(MappingDesc *cellDescription)
        {
        }

    protected:

        virtual void moduleLoad()
        {
        }

        virtual void moduleUnload()
        {
        }
    };

}

#endif	/* INITMODULENONE_HPP */

