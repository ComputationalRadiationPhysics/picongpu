/**
 * Copyright 2013 Heiko Burau, Rene Widera
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
 
#ifndef ANALYSIS_TOTALDIVJ_HPP
#define ANALYSIS_TOTALDIVJ_HPP

#include "plugins/ILightweightPlugin.hpp"

namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;

class TotalDivJ : public ILightweightPlugin
{
private:
    std::string name;
    std::string prefix;
    uint32_t notifyFrequency;
        
    void pluginLoad();
public:
    TotalDivJ(std::string name, std::string prefix);
    virtual ~TotalDivJ() {}

    void notify(uint32_t currentStep);
    void setMappingDescription(MappingDesc*) {}
    void pluginRegisterHelp(po::options_description& desc);
    std::string pluginGetName() const;
};

}

#include "TotalDivJ.tpp"

#endif // ANALYSIS_TOTALDIVJ_HPP
