/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Rene Widera, Richard Pausch
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
 


#ifndef COUNTPARTICLES_HPP
#define	COUNTPARTICLES_HPP

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include "simulation_classTypes.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "moduleSystem/Module.hpp"
#include "plugins/IPluginModule.hpp"

#include "mpi/reduceMethods/Reduce.hpp"
#include "mpi/MPIReduce.hpp"
#include "nvidia/functors/Add.hpp"
#include "nvidia/functors/Max.hpp"

#include "simulation_classTypes.hpp"

#include "particles/operations/CountParticles.hpp"

namespace picongpu
{
using namespace PMacc;

template<class ParticlesType>
class CountParticles : public ISimulationIO, public IPluginModule
{
private:
    typedef MappingDesc::SuperCellSize SuperCellSize;

    ParticlesType *particles;

    MappingDesc *cellDescription;
    uint32_t notifyFrequency;

    std::string analyzerName;
    std::string analyzerPrefix;
    std::string filename;

    std::ofstream outFile;
    /*only rank 0 create a file*/
    bool writeToFile;

    mpi::MPIReduce reduce;
public:

    CountParticles(std::string name, std::string prefix) :
    analyzerName(name),
    analyzerPrefix(prefix),
    filename(name + ".dat"),
    particles(NULL),
    cellDescription(NULL),
    notifyFrequency(0),
    writeToFile(false)
    {
        ModuleConnector::getInstance().registerModule(this);
    }

    virtual ~CountParticles()
    {

    }

    void notify(uint32_t currentStep)
    {
        DataConnector &dc = DataConnector::getInstance();

        particles = &(dc.getData<ParticlesType > ((uint32_t) ParticlesType::FrameType::CommunicationTag, true));

        countParticles < CORE + BORDER > (currentStep);
    }

    void moduleRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ((analyzerPrefix + ".period").c_str(),
             po::value<uint32_t > (&notifyFrequency), "enable plugin [for each n-th step]");
    }

    std::string moduleGetName() const
    {
        return analyzerName;
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

private:

    void moduleLoad()
    {
        if (notifyFrequency > 0)
        {
            writeToFile = reduce.hasResult(mpi::reduceMethods::Reduce());

            if (writeToFile)
            {
                outFile.open(filename.c_str(), std::ofstream::out | std::ostream::trunc);
                if (!outFile)
                {
                    std::cerr << "Can't open file [" << filename << "] for output, disable plugin output. " << std::endl;
                    writeToFile = false;
                }
                //create header of the file
                outFile << "#step count" << " \n";
            }

            DataConnector::getInstance().registerObserver(this, notifyFrequency);
        }
    }

    void moduleUnload()
    {
        if (notifyFrequency > 0)
        {
            if (writeToFile)
            {
                outFile.flush();
                outFile << std::endl; //now all data are written to file
                if (outFile.fail())
                    std::cerr << "Error on flushing file [" << filename << "]. " << std::endl;
                outFile.close();
            }
        }
    }

    template< uint32_t AREA>
    void countParticles(uint32_t currentStep)
    {
        uint64_cu size;

        PMACC_AUTO(simBox, SubGrid<simDim>::getInstance().getSimulationBox());
        const DataSpace<simDim> localSize(simBox.getLocalSize());

        /*count local particles*/
        size = PMacc::CountParticles::countOnDevice<AREA>(*particles,
                                                          *cellDescription,
                                                          DataSpace<simDim>(),
                                                          localSize);
        uint64_cu reducedValueMax;
        if (picLog::log_level & picLog::CRITICAL::lvl)
        {
            reduce(nvidia::functors::Max(),
                   &reducedValueMax,
                   &size,
                   1,
                   mpi::reduceMethods::Reduce());
        }


        uint64_cu reducedValue;
        reduce(nvidia::functors::Add(),
               &reducedValue,
               &size,
               1,
               mpi::reduceMethods::Reduce());
        
        if (writeToFile)
        {
            if (picLog::log_level & picLog::CRITICAL::lvl)
            {
                log<picLog::CRITICAL > ("maximum number of  particles on a GPU : %d\n") % reducedValueMax;
            }

            outFile << currentStep << " " << reducedValue << " " << std::scientific << (double) reducedValue << std::endl;
        }
    }

};

}

#endif	/* COUNTPARTICLES_HPP */

