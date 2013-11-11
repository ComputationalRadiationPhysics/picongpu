/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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
 


#ifndef ENERGYFIELDS_HPP
#define	ENERGYFIELDS_HPP

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

namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;

typedef typename FieldB::DataBoxType B_DataBox;
typedef typename FieldE::DataBoxType E_DataBox;

template<class Mapping>
__global__ void kernelEnergy(E_DataBox fieldE, B_DataBox fieldB, float* gEnergy, Mapping mapper)
{

    //__shared__ float_X sh_sumE2;
    //__shared__ float_X sh_sumB2;
    __shared__ float_X sh_sumEn;

    __syncthreads(); /*wait that all shared memory is initialised*/

    typedef typename Mapping::SuperCellSize SuperCellSize;
    const DataSpace<simDim > threadIndex(threadIdx);
    const int linearThreadIdx = DataSpaceOperations<simDim>::template map<SuperCellSize > (
                                                                                           threadIndex
                                                                                           );

    if (linearThreadIdx == 0)
    {
        //sh_sumE2 = float_X(0.0);
        //sh_sumB2 = float_X(0.0);
        sh_sumEn = float_X(0.0);
    }

    __syncthreads();

    const DataSpace<simDim> superCellIdx(mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx)));
    const DataSpace<simDim> cell(superCellIdx * SuperCellSize() + threadIndex);

    const float3_X b = fieldB(cell);
    const float3_X e = fieldE(cell);

    const float_X myE2 = e.x() * e.x() + e.y() * e.y() + e.z() * e.z();
    const float_X myB2 = b.x() * b.x() + b.y() * b.y() + b.z() * b.z();

    const float_X volume = CELL_WIDTH * CELL_HEIGHT * CELL_DEPTH;
    const float_X myEn = ((EPS0 * myE2) + (myB2 * (float_X(1.0) / MUE0))) * (volume * float_X(0.5));

    //atomicAddWrapper(&sh_sumE2, myE2);
    //atomicAddWrapper(&sh_sumB2, myB2);
    atomicAddWrapper(&sh_sumEn, myEn);

    __syncthreads();

    if (linearThreadIdx == 0)
    {
        //const float_X volume = double(MappingDesc::SuperCellSize::elements) * CELL_WIDTH * CELL_HEIGHT * CELL_DEPTH;
        //double globalEnergy = ((double) (EPS0 * sh_sumE2) + (double) (sh_sumB2 * (float_X(1.0) / MUE0))) * (double) (volume * float_X(0.5));
        //atomicAddWrapper(gEnergy, globalEnergy);

        //atomicAddWrapper(&gEnergy[0], double(sh_sumE2));
        //atomicAddWrapper(&gEnergy[1], double(sh_sumB2));

        atomicAddWrapper(&gEnergy[0], sh_sumEn);
    }
}

class EnergyFields : public ISimulationIO, public IPluginModule
{
private:
    FieldE* fieldE;
    FieldB* fieldB;

    MappingDesc *cellDescription;
    uint32_t notifyFrequency;
    GridBuffer<float, DIM1> *energy;

    std::string analyzerName;
    std::string analyzerPrefix;
    std::string filename;
    std::ofstream outFile;
    /*only rank 0 create a file*/
    bool writeToFile;

    mpi::MPIReduce reduce;


public:

    EnergyFields(std::string name, std::string prefix) :
    fieldE(NULL),
    fieldB(NULL),
    cellDescription(NULL),
    analyzerName(name),
    analyzerPrefix(prefix),
    filename(name + ".dat"),
    notifyFrequency(0),
    writeToFile(false)
    {


        ModuleConnector::getInstance().registerModule(this);
    }

    virtual ~EnergyFields()
    {

    }

    void notify(uint32_t currentStep)
    {
        DataConnector &dc = DataConnector::getInstance();

        fieldE = &(dc.getData<FieldE > (FIELD_E, true));
        fieldB = &(dc.getData<FieldB > (FIELD_B, true));
        getEnergyFields < CORE + BORDER > (currentStep);
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
            writeToFile = reduce.hasResult(mpi::reduceMethods::Reduce());
            energy = new GridBuffer<float, DIM1 > (DataSpace<DIM1 > (1)); //create one int on gpu und host

            if (writeToFile)
            {
                outFile.open(filename.c_str(), std::ofstream::out | std::ostream::trunc);
                if (!outFile)
                {
                    std::cerr << "Can't open file [" << filename << "] for output, diasble analyser output. " << std::endl;
                    writeToFile = false;
                }
                //create header of the file
                outFile << "#step Joule" << " \n";
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
            if (energy)
                delete energy;
        }
    }

    template< uint32_t AREA>
    void getEnergyFields(uint32_t currentStep)
    {
        energy->getDeviceBuffer().setValue(0.0);
        dim3 block(MappingDesc::SuperCellSize::getDataSpace());

        __picKernelArea(kernelEnergy, *cellDescription, AREA)
            (block)
            (fieldE->getDeviceDataBox(),
             fieldB->getDeviceDataBox(),
             energy->getDeviceBuffer().getBasePointer());
        energy->deviceToHost();

        double globalEnergy = double( energy->getHostBuffer().getBasePointer()[0]);
        double reducedValue;
        reduce(nvidia::functors::Add(),
               &reducedValue,
               &globalEnergy,
               1,
               mpi::reduceMethods::Reduce());

        if (writeToFile)
        {
            typedef std::numeric_limits< float_64 > dbl;

            outFile.precision(dbl::digits10);
            outFile << currentStep << " " << std::scientific << reducedValue * UNIT_ENERGY << std::endl;
        }
    }

};

}

#endif	/* ENERGYFIELDS_HPP */

