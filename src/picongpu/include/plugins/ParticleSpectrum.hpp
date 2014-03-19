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
 
#ifndef ANALYSIS_PARTICLESPECTRUM_HPP
#define ANALYSIS_PARTICLESPECTRUM_HPP


#include "plugins/ISimulationPlugin.hpp"

namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;

template<typename ParticlesType>
class ParticleSpectrum : public ISimulationPlugin
{
private:
    std::string name;
    std::string prefix;
    uint32_t notifyFrequency;
    float_X minEnergy, maxEnergy;
    static const int numBins = 64;
    static const int numBinsEx = numBins+2;
    ParticlesType *particles;
        
    void pluginLoad();
    void pluginUnload();
public:
    ParticleSpectrum(std::string name, std::string prefix);
    ~ParticleSpectrum() {}

    void notify(uint32_t currentStep);
    void setMappingDescription(MappingDesc*) {}
    void pluginRegisterHelp(po::options_description& desc);
    std::string pluginGetName() const;
};

namespace detail
{
template<int _numBins>
struct Histrogram
{
    static const int numBins = _numBins;
    static const int numBinsEx = numBins + 2;
    float_X bin[numBinsEx];
};
}

}

#include "ParticleSpectrum.tpp"

#endif // ANALYSIS_PARTICLESPECTRUM_HPP
