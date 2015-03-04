/**
 * Copyright 2013-2015 Anshuman Goswami
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

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"
#include "basicOperations.hpp"
#include "dimensions/DataSpace.hpp"

#include "simulation_classTypes.hpp"
#include "mappings/kernel/AreaMapping.hpp"
#include "plugins/ISimulationPlugin.hpp"

#include "mpi/reduceMethods/Reduce.hpp"
#include "mpi/MPIReduce.hpp"
#include "nvidia/functors/Add.hpp"

#include "algorithms/Gamma.hpp"

#include <boost/filesystem.hpp>

namespace picongpu
{
using namespace PMacc;
using namespace boost::filesystem;

namespace po = boost::program_options;

template<class ParticlesType>
class InsituBinEnergyParticles : public ISimulationPlugin
{
private:

    ParticlesType *particles;

    int numBins;
    int realNumBins;

public:

    InsituBinEnergyParticles() :
    particles(NULL),
    {
    }

    virtual ~InsituBinEnergyParticles()
    {

    }

    void notify(uint32_t currentStep)
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        particles = &(dc.getData<ParticlesType > (ParticlesType::FrameType::getName(), true));

	//Serialize particles::superCells

        //Submit serialized particle data to the transport library
    }

    void pluginRegisterHelp(po::options_description& desc)
    {
	//No cmdline interface now
    }

    std::string pluginGetName() const
    {
        return NULL;
    }

    void pluginLoad()
    {
	    //Also setup transport library
    }

    void pluginUnload()
    {
	    //Also teardown transport library
    }

    void restart(uint32_t restartStep, const std::string restartDirectory)
    {
	//Not implemented
    }

    void checkpoint(uint32_t currentStep, const std::string checkpointDirectory)
    {
	//Not implemented
    }

    //calBinEnergyParticles is now run in a different address space
};

}
