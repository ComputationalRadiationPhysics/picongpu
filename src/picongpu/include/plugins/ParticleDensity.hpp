/**
 * Copyright 2013-2014 Heiko Burau, Rene Widera, Felix Schmitt
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


#include "plugins/ILightweightPlugin.hpp"

namespace picongpu
{
namespace heiko
{
using namespace PMacc;

namespace po = boost::program_options;

template<typename ParticlesType>
class ParticleDensity : public ILightweightPlugin
{
private:
    std::string name;
    std::string prefix;
    uint32_t notifyFrequency;
    float_X factor;
    int plane;
    float_X slicePoint;
    ParticlesType *particles;

public:
    ParticleDensity(std::string name, std::string prefix);
    ~ParticleDensity() {}

    void notify(uint32_t currentStep);
    void setMappingDescription(MappingDesc*) {}
    void pluginRegisterHelp(po::options_description& desc);
    std::string pluginGetName() const;
    
private:
    void pluginLoad();
};

}
}

#include "ParticleDensity.tpp"
