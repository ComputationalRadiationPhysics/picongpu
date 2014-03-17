/**
 * Copyright 2013-2014 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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
#include "plugins/ISimulationPlugin.hpp"

#include "mpi/reduceMethods/Reduce.hpp"
#include "mpi/MPIReduce.hpp"
#include "nvidia/functors/Add.hpp"
#include "nvidia/reduce/Reduce.hpp"
#include "memory/boxes/DataBoxDim1Access.hpp"
#include "memory/boxes/DataBoxUnaryTransform.hpp"

namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;

namespace energyFields
{

template<typename T_Type>
struct cast64Bit
{
    typedef float_64 result;

    HDINLINE typename TypeCast<result, T_Type>::result operator()(const T_Type& value) const
    {
        return precisionCast<result>(value);
    }
};
}

class EnergyFields : public ISimulationIO, public ISimulationPlugin
{
private:
    FieldE* fieldE;
    FieldB* fieldB;

    MappingDesc *cellDescription;
    uint32_t notifyFrequency;

    std::string analyzerName;
    std::string analyzerPrefix;
    std::string filename;
    std::ofstream outFile;
    /*only rank 0 create a file*/
    bool writeToFile;

    mpi::MPIReduce mpiReduce;

    nvidia::reduce::Reduce* localReduce;

public:

    EnergyFields(std::string name, std::string prefix) :
    fieldE(NULL),
    fieldB(NULL),
    cellDescription(NULL),
    analyzerName(name),
    analyzerPrefix(prefix),
    filename(name + ".dat"),
    notifyFrequency(0),
    writeToFile(false),
    localReduce(NULL)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~EnergyFields()
    {

    }

    void notify(uint32_t currentStep)
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        fieldE = &(dc.getData<FieldE > (FieldE::getName(), true));
        fieldB = &(dc.getData<FieldB > (FieldB::getName(), true));
        getEnergyFields(currentStep);
    }

    void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ((analyzerPrefix + ".period").c_str(),
             po::value<uint32_t > (&notifyFrequency)->default_value(0), "enable analyser [for each n-th step]");
    }

    std::string pluginGetName() const
    {
        return analyzerName;
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

private:

    void pluginLoad()
    {
        if (notifyFrequency > 0)
        {
            localReduce = new nvidia::reduce::Reduce(1024);
            writeToFile = mpiReduce.hasResult(mpi::reduceMethods::Reduce());

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
            Environment<>::get().DataConnector().registerObserver(this, notifyFrequency);
        }
    }

    void pluginUnload()
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
            __delete(localReduce);
        }
    }

    void getEnergyFields(uint32_t currentStep)
    {
        float_64 fieldEReduced = reduceField(fieldE);
        float_64 fieldBReduced = reduceField(fieldB);

        float_64 localFieldEnergy = ((EPS0 * fieldEReduced) + (fieldBReduced * (float_X(1.0) / MUE0))) * (CELL_VOLUME * float_X(0.5));
        float_64 globalEnergy = 0.0;

        mpiReduce(nvidia::functors::Add(),
                  &globalEnergy,
                  &localFieldEnergy,
                  1,
                  mpi::reduceMethods::Reduce());

        if (writeToFile)
        {
            typedef std::numeric_limits< float_64 > dbl;

            outFile.precision(dbl::digits10);
            outFile << currentStep << " " << std::scientific << globalEnergy * UNIT_ENERGY << std::endl;
        }
    }

private:

    template<typename T_Field>
    float_64 reduceField(T_Field* field)
    {
        /*define stacked DataBox's for reduce algorithm*/
        typedef DataBoxUnaryTransform<typename T_Field::DataBoxType, math::Abs2 > TransformedBox;
        typedef DataBoxUnaryTransform<TransformedBox, energyFields::cast64Bit > Box64bit;
        typedef DataBoxDim1Access<Box64bit > D1Box;

        /* reduce field E*/
        DataSpace<simDim> fieldSize = field->getGridLayout().getDataSpaceWithoutGuarding();
        DataSpace<simDim> fieldGuard = field->getGridLayout().getGuard();

        TransformedBox fieldTransform(field->getDeviceDataBox().shift(fieldGuard));
        Box64bit field64bit(fieldTransform);
        D1Box d1Access(field64bit, fieldSize);

        float_64 fieldReduced = (*localReduce)(nvidia::functors::Add(),
                                               d1Access,
                                               fieldSize.productOfComponents());

        return fieldReduced;
    }

};

} //namespace picongpu
