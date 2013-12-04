/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Heiko Burau, René Widera
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

#include <iostream>
#include <fstream>

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include "simulation_classTypes.hpp"

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"

#include "basicOperations.hpp"
#include "dimensions/DataSpaceOperations.hpp"
#include "plugins/IPluginModule.hpp"

#include "mpi/reduceMethods/Reduce.hpp"
#include "mpi/MPIReduce.hpp"
#include "nvidia/functors/Add.hpp"

#include "cuSTL/container/DeviceBuffer.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/algorithm/kernel/Reduce.hpp"
#include "lambda/Expression.hpp"
#include "math/vector/compile-time/Int.hpp"

namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;

//typedef typename FieldB::DataBoxType B_DataBox;
//typedef typename FieldE::DataBoxType E_DataBox;

class HelloWorld : public ISimulationIO, public IPluginModule
{
private:
//    FieldE* fieldE;
//    FieldB* fieldB;

    MappingDesc *cellDescription;
    uint32_t notifyFrequency;
    //GridBuffer<float, DIM1> *energy;
    GridBuffer<float, DIM1> *number;

    std::string analyzerName;
    std::string analyzerPrefix;
    std::string filename;
    std::ofstream outFile;
    /*only rank 0 create a file*/
    bool writeToFile;

    mpi::MPIReduce reduce;


public:

    HelloWorld(std::string name, std::string prefix) :
//    fieldE(NULL),
//    fieldB(NULL),
    cellDescription(NULL),
    analyzerName(name),
    analyzerPrefix(prefix),
    filename(name + ".dat"),
    notifyFrequency(0),
    writeToFile(false)
    {


        ModuleConnector::getInstance().registerModule(this);
    }

    virtual ~HelloWorld()
    {

    }

    void notify(uint32_t currentStep)
    {   
        
        std::cout << "##########  PLUGIN START  ###########" << std::endl;
        // create device Buffer
        container::DeviceBuffer<int, 2> deviceBuffer(16, 16);
        
        // fill with ones
        deviceBuffer.assign(1);
        
//        // create result container
//        container::DeviceBuffer<int, 1> deviceResult(1);
//        
//        using namespace lambda;
//        // reduce algorithm
//        PMacc::algorithm::kernel::Reduce<PMacc::math::CT::Int<4, 4, 1> > reduce;
//        reduce(deviceResult.origin(), deviceBuffer.zone(), deviceBuffer.origin(), _1 + _2);
//        
//        // create host Buffer
//        container::HostBuffer<int, 1> hostResult(1);
//        
//        // copy result from device to host
//        hostResult = deviceResult;
//        
//        // print result
//        std::cout << hostResult << std::endl;
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



