/* 
 * File:   GetAttributes.hpp
 * Author: garten70
 *
 * Created on 12. Februar 2014, 13:51
 */

#pragma once

#include <iostream>
#include <fstream>

#include "types.h"
#include "particles/frame_types.hpp"
#include "particles/memory/boxes/ParticlesBox.hpp"
#include "particles/memory/boxes/TileDataBox.hpp"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include "simulation_classTypes.hpp"

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"

#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/CachedBox.hpp"

#include "basicOperations.hpp"
#include "dimensions/DataSpaceOperations.hpp"
#include "plugins/IPluginModule.hpp"

#include "mpi/reduceMethods/Reduce.hpp"
#include "mpi/MPIReduce.hpp"
#include "nvidia/functors/Add.hpp"
#include "particles/operations/Assign.hpp"
#include "particles/operations/Deselect.hpp"


#include "cuSTL/container/DeviceBuffer.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/algorithm/kernel/Reduce.hpp"
#include "lambda/Expression.hpp"
#include "math/vector/compile-time/Int.hpp"


namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;

class GetAttributes : public ISimulationIO, public IPluginModule
{
private:

    MappingDesc *cellDescription;
    uint32_t notifyFrequency;
    GridBuffer<float, DIM1> *number;

    std::string analyzerName;
    std::string analyzerPrefix;
    std::string filename;
    std::ofstream outFile;
    /*only rank 0 create a file*/
    bool writeToFile;

    mpi::MPIReduce reduce;


public:

    GetAttributes(std::string name, std::string prefix) :
    cellDescription(NULL),
    analyzerName(name),
    analyzerPrefix(prefix),
    filename(name + ".dat"),
    notifyFrequency(0),
    writeToFile(false)
    {
        ModuleConnector::getInstance().registerModule(this);
    }

    virtual ~GetAttributes()
    {

    }

    void notify(uint32_t currentStep)
    {   
        
        std::cout << "##########  PLUGIN START  ###########" << std::endl;

    }

    void moduleRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ((analyzerPrefix + ".period").c_str(),
             po::value<uint32_t > (&notifyFrequency)->default_value(0), "enable analyser [for each n-th step]");
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
            DataConnector::getInstance().registerObserver(this, notifyFrequency);
        }
    }

    void moduleUnload()
    {
        
    }

};

}

