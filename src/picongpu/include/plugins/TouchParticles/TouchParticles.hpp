/**
 * Copyright 2013 Heiko Burau, René Widera
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

#include "plugins/IPluginModule.hpp"
#include "dataManagement/ISimulationIO.hpp"
#include "simulation_classTypes.hpp" //\todo: muss in ISimulationIO.hpp

namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;

class TouchParticles : public ISimulationIO, public IPluginModule
{
private:
    std::string name;
    std::string prefix;
    uint32_t notifyFrequency;
        
    void moduleLoad();
    void moduleUnload();
public:
    TouchParticles(std::string name, std::string prefix);
    virtual ~TouchParticles() {}

    void notify(uint32_t currentStep);
    void setMappingDescription(MappingDesc*) {}
    void moduleRegisterHelp(po::options_description& desc);
    std::string moduleGetName() const;
};

}

#include "TouchParticles.tpp"

