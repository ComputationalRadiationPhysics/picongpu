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
 


#ifndef ENERGYPARTICLES_HPP
#define	ENERGYPARTICLES_HPP

#include <string>
#include <iostream>
#include <fstream>
#include <mpi.h>

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"
#include "basicOperations.hpp"

#include "simulation_classTypes.hpp"
#include "mappings/kernel/AreaMapping.hpp"
#include "plugins/IPluginModule.hpp"

#include "mpi/reduceMethods/Reduce.hpp"
#include "mpi/MPIReduce.hpp"
#include "nvidia/functors/Add.hpp"

#include "algorithms/Gamma.hpp"

namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;

/* sum up the energy of all particles
 * the kinetic energy of all active particles will be calculated
 */
template<class FRAME, class DBox, class Mapping>
__global__ void kernelEnergyParticles(ParticlesBox<FRAME, simDim> pb,
                                      DBox gEnergy,
                                      Mapping mapper)
{

    __shared__ FRAME *frame;
    __shared__ bool isValid;
    __shared__ float_X shEnergyKin;
    __shared__ float_X shEnergy;
    __syncthreads(); /*wait that all shared memory is initialised*/

    float_X _local_energyKin = float_X(0.0);
    float_X _local_energy = float_X(0.0);


    typedef typename Mapping::SuperCellSize SuperCellSize;

    const DataSpace<simDim > threadIndex(threadIdx);
    const int linearThreadIdx = DataSpaceOperations<simDim>::template map<SuperCellSize > (threadIndex);

    if (linearThreadIdx == 0)
    {
        const DataSpace<simDim> superCellIdx(mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx)));
        frame = &(pb.getLastFrame(superCellIdx, isValid));
        shEnergyKin = float_X(0.0);
        shEnergy = float_X(0.0);
    }

    __syncthreads();
    if (!isValid)
        return; //end kernel if we have no frames

    bool isParticle = (*frame)[linearThreadIdx][multiMask_];

    while (isValid)
    {
        if (isParticle)
        {

            PMACC_AUTO(particle,(*frame)[linearThreadIdx]);
            const float3_X mom = particle[momentum_];
            const float_X mom2 = mom.x() * mom.x() + mom.y() * mom.y() + mom.z() * mom.z();

            const float_X weighting = particle[weighting_];
            const float_X mass = frame->getMass(weighting);
            const float_X c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

            Gamma<> calcGamma;
            const float_X gamma = calcGamma(mom, mass);

            if (gamma < 1.005f)
            {
                _local_energyKin += mom2 / (2.0f * mass); //not relativistic use equation with more precision
            }
            else
            {
                // kinetic Energy for Particles: E = (gamma - 1) * m * c^2
                //                                    gamma = sqrt( 1 + (p/m/c)^2 )
                //_local_energyKin += (sqrtf(mom2 / (mass * mass * c2) + 1.) - 1.) * mass * c2;

                _local_energyKin += (gamma - float_X(1.0)) * mass*c2;
            }

            // kinetic Energy for Particles: E^2 = p^2*c^2 + m^2*c^4
            //                                   = c^2 * [p^2 + m^2*c^2]
            _local_energy += sqrtf(mom2 + mass * mass * c2) * SPEED_OF_LIGHT;

        }
        __syncthreads();
        if (linearThreadIdx == 0)
        {
            frame = &(pb.getPreviousFrame(*frame, isValid));
        }
        isParticle = true;
        __syncthreads();
    }

    atomicAddWrapper(&shEnergyKin, _local_energyKin);
    atomicAddWrapper(&shEnergy, _local_energy);

    __syncthreads();
    if (linearThreadIdx == 0)
    {
        // kinetic Energy for Particles: E^2 = p^2*c^2 + m^2*c^4
        //                                   = c^2 * [p^2 + m^2*c^2]
        atomicAddWrapper(&(gEnergy[0]), (float_64) (shEnergyKin));
        atomicAddWrapper(&(gEnergy[1]), (float_64) (shEnergy));
    }
}

template<class ParticlesType>
class EnergyParticles : public ISimulationIO, public IPluginModule
{
private:
    typedef MappingDesc::SuperCellSize SuperCellSize;

    ParticlesType *particles;


    GridBuffer<double, DIM1> *gEnergy;
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

    EnergyParticles(std::string name, std::string prefix) :
    analyzerName(name),
    analyzerPrefix(prefix),
    filename(name + ".dat"),
    particles(NULL),
    gEnergy(NULL),
    cellDescription(NULL),
    notifyFrequency(0),
    writeToFile(false)
    {
        ModuleConnector::getInstance().registerModule(this);
    }

    virtual ~EnergyParticles()
    {

    }

    void notify(uint32_t currentStep)
    {
        DataConnector &dc = DataConnector::getInstance();

        particles = &(dc.getData<ParticlesType > ((uint32_t) ParticlesType::FrameType::CommunicationTag, true));

        calculateEnergyParticles < CORE + BORDER > (currentStep);
    }

    void moduleRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ((analyzerPrefix + ".period").c_str(),
             po::value<uint32_t > (&notifyFrequency), "enable analyser [for each n-th step]");
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
            gEnergy = new GridBuffer<double, DIM1 > (DataSpace<DIM1 > (2)); //create one int on gpu und host

            if (writeToFile)
            {
                outFile.open(filename.c_str(), std::ofstream::out | std::ostream::trunc);
                if (!outFile)
                {
                    std::cerr << "Can't open file [" << filename << "] for output, diasble analyser output. " << std::endl;
                    writeToFile = false;
                }
                //create header of the file
                outFile << "#step Ekin_Joule E_Joule" << " \n";
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

            __delete(gEnergy);
        }
    }

    template< uint32_t AREA>
    void calculateEnergyParticles(uint32_t currentStep)
    {
        gEnergy->getDeviceBuffer().setValue(0.0);
        dim3 block(MappingDesc::SuperCellSize::getDataSpace());

        __picKernelArea(kernelEnergyParticles, *cellDescription, AREA)
            (block)
            (particles->getDeviceParticlesBox(),
             gEnergy->getDeviceBuffer().getDataBox());
        gEnergy->deviceToHost();


        double reducedEnergy[2];

        reduce(nvidia::functors::Add(),
               reducedEnergy,
               gEnergy->getHostBuffer().getBasePointer(),
               2,
               mpi::reduceMethods::Reduce());

        if (writeToFile)
        {
            typedef std::numeric_limits< float_64 > dbl;

            outFile.precision(dbl::digits10);
            outFile << currentStep << " " << std::scientific << reducedEnergy[0] * UNIT_ENERGY << " " <<
                reducedEnergy[1] * UNIT_ENERGY << std::endl;
        }
    }

};

}

#endif	/* ENERGYPARTICLES_HPP */

