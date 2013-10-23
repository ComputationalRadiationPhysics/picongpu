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
 


#ifndef BINENERGYPARTICLES_HPP
#define	BINENERGYPARTICLES_HPP

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"
#include "basicOperations.hpp"
#include "dimensions/DataSpace.hpp"

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
template<class FRAME, class BinBox, class Mapping>
__global__ void kernelBinEnergyParticles(ParticlesBox<FRAME, simDim> pb,
                                         BinBox gBins, int numBins,
                                         float_X minEnergy,
                                         float_X maxEnergy,
                                         float_X maximumSlopeToDetectorX,
                                         float_X maximumSlopeToDetectorZ,
                                         Mapping mapper)
{

    typedef typename MappingDesc::SuperCellSize Block;

    __shared__ FRAME *frame;
    __shared__ bool isValid;
    __shared__ lcellId_t particlesInSuperCell;

    const bool enableDetector = maximumSlopeToDetectorX != float_X(0.0) && maximumSlopeToDetectorZ != float_X(0.0);

    /* shBins index can go from 0 to (numBins+2)-1
     * 0 is for <minEnergy
     * (numBins+2)-1 is for >maxEnergy
     */
    extern __shared__ float_X shBin[]; //size must be numBins+2 because we have <min and >max

    __syncthreads(); /*wait that all shared memory is initialised*/

    int realNumBins = numBins + 2;



    typedef typename Mapping::SuperCellSize SuperCellSize;
    const int threads = SuperCellSize::elements;

    const DataSpace<simDim > threadIndex(threadIdx);
    const int linearThreadIdx = DataSpaceOperations<simDim>::template map<SuperCellSize > (threadIndex);

    if (linearThreadIdx == 0)
    {
        const DataSpace<simDim> superCellIdx(mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx)));
        frame = &(pb.getLastFrame(superCellIdx, isValid));
        particlesInSuperCell = pb.getSuperCell(superCellIdx).getSizeLastFrame();
    }
    //set all bins to 0
    for (int i = linearThreadIdx; i < realNumBins; i += threads)
    {
        shBin[i] = float_X(0.);
    }

    __syncthreads();
    if (!isValid)
        return; //end kernel if we have no frames

    // bool isParticle = frame->getMultiMask()[linearThreadIdx];

    while (isValid)
    {
        if (linearThreadIdx < particlesInSuperCell)
        {
            PMACC_AUTO(particle,(*frame)[linearThreadIdx]);
            // kinetic Energy for Particles: E^2 = p^2*c^2 + m^2*c^4
            //                                   = c^2 * [p^2 + m^2*c^2]
            const float3_X mom = particle[momentum_];

            bool calcParticle = true;

            if (enableDetector && mom.y() > 0.0)
            {
                const float_X slopeMomX = abs(mom.x() / mom.y());
                const float_X slopeMomZ = abs(mom.z() / mom.y());
                if (slopeMomX >= maximumSlopeToDetectorX || slopeMomZ >= maximumSlopeToDetectorZ)
                {
                    calcParticle = false;
                }
            }

            if (calcParticle)
            {
                const float_X mom2 = math::abs2(mom);
                const float_X weighting = particle[weighting_];
                const float_X mass = frame->getMass(weighting);
                const float_X c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

                Gamma<> calcGamma;
                const float_X gamma = calcGamma(mom, mass);

                float_X _local_energy;

                if (gamma < 1.005f)
                {
                    _local_energy = mom2 / (2.0f * mass); //not relativistic use equation with more precision
                }
                else
                {
                    // kinetic Energy for Particles: E = (sqrt[p^2*c^2 /(m^2*c^4)+ 1] -1) m*c^2
                    //                                   = c^2 * [p^2 + m^2*c^2]-m*c^2
                    //                                 = (gamma - 1) * m * c^2
                    _local_energy = (gamma - float_X(1.0)) * mass*c2;
                }
                _local_energy /= weighting;

                /*+1 move value from 1 to numBins+1*/
                int binNumber = math::floor((_local_energy - minEnergy) /
                                      (maxEnergy - minEnergy) * (float) numBins) + 1;

                const int maxBin = numBins + 1;
                /* same as
                 * binNumber<numBins?binNumber:numBins+1
                 * but without if*/
                binNumber = binNumber * (int) (binNumber < maxBin) +
                    maxBin * (int) (binNumber >= maxBin);
                /* same as
                 * binNumber>0?binNumber:0
                 * but without if*/
                binNumber = (int) (binNumber > 0) * binNumber;

                /*!\todo: we can't use 64bit type on this place (NVIDIA BUG?)
                 * COMPILER ERROR: ptxas /tmp/tmpxft_00005da6_00000000-2_main.ptx, line 4246; error   : Global state space expected for instruction 'atom'
                 * I think this is a problem with extern shared mem and atmic (only on TESLA)
                 * NEXT BUG: don't do uint32_t w=__float2uint_rn(weighting); and use w for atomic, this create wrong results
                 */
                // overflow for big weighting reduces in shared mem
                //atomicAdd(&(shBin[binNumber]), (uint32_t) weighting);
                const float_X normedWeighting = float_X(weighting) / float_X(NUM_EL_PER_PARTICLE);
                atomicAddWrapper(&(shBin[binNumber]), normedWeighting);
            }
        }
        __syncthreads();
        if (linearThreadIdx == 0)
        {
            frame = &(pb.getPreviousFrame(*frame, isValid));
            particlesInSuperCell = Block::elements;
        }
        __syncthreads();
    }

    __syncthreads();
    for (int i = linearThreadIdx; i < realNumBins; i += threads)
    {
        atomicAddWrapper(&(gBins[i]), float_64(shBin[i]));
    }
    __syncthreads();
}

template<class ParticlesType>
class BinEnergyParticles : public ISimulationIO, public IPluginModule
{
private:

    typedef MappingDesc::SuperCellSize SuperCellSize;

    ParticlesType *particles;

    GridBuffer<double, DIM1> *gBins;
    MappingDesc *cellDescription;

    std::string analyzerName;
    std::string analyzerPrefix;
    std::string filename;

    double * binReduced;

    uint32_t notifyFrequency;
    int numBins;
    int realNumBins;
    /*in picongpu units*/
    float_X minEnergy;
    float_X maxEnergy;

    float_X distanceToDetector;
    float_X slitDetectorX;
    float_X slitDetectorZ;
    bool enableDetector;

    std::ofstream outFile;

    /*only rank 0 create a file*/
    bool writeToFile;

    mpi::MPIReduce reduce;

public:

    BinEnergyParticles(std::string name, std::string prefix) :
    analyzerName(name),
    analyzerPrefix(prefix),
    filename(name + ".dat"),
    particles(NULL),
    gBins(NULL),
    cellDescription(NULL),
    notifyFrequency(0),
    writeToFile(false),
    enableDetector(false)
    {
        ModuleConnector::getInstance().registerModule(this);
    }

    virtual ~BinEnergyParticles()
    {

    }

    void notify(uint32_t currentStep)
    {
        DataConnector &dc = DataConnector::getInstance();

        particles = &(dc.getData<ParticlesType > ((uint32_t) ParticlesType::FrameType::CommunicationTag, true));
        calBinEnergyParticles < CORE + BORDER > (currentStep);
    }

    void moduleRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ((analyzerPrefix + ".period").c_str(), po::value<uint32_t > (&notifyFrequency)->default_value(0), "enable analyser [for each n-th step]")
            ((analyzerPrefix + ".binCount").c_str(), po::value<int > (&numBins)->default_value(1024), "binCount")
            ((analyzerPrefix + ".minEnergy").c_str(), po::value<float_X > (&minEnergy)->default_value(0.0), "minEnergy[in keV]")
            ((analyzerPrefix + ".maxEnergy").c_str(), po::value<float_X > (&maxEnergy), "maxEnergy[in keV]")
            ((analyzerPrefix + ".distanceToDetector").c_str(), po::value<float_X > (&distanceToDetector)->default_value(0.0), "distance between gas and detector, assumptions: simulated area in y direction << distance to detector AND simulated area in X,Z << slit [in meters]  (if not set, all particles are counted)")
            ((analyzerPrefix + ".slitDetectorX").c_str(), po::value<float_X > (&slitDetectorX)->default_value(0.0), "size of the detector slit in X [in meters] (if not set, all particles are counted)")
            ((analyzerPrefix + ".slitDetectorZ").c_str(), po::value<float_X > (&slitDetectorZ)->default_value(0.0), "size of the detector slit in Z [in meters] (if not set, all particles are counted)");
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


            if (distanceToDetector != float_X(0.0) && slitDetectorX != float_X(0.0) && slitDetectorZ != float_X(0.0))
                enableDetector = true;

            realNumBins = numBins + 2;
            minEnergy = minEnergy * UNITCONV_keV_to_Joule / UNIT_ENERGY;
            maxEnergy = maxEnergy * UNITCONV_keV_to_Joule / UNIT_ENERGY;

            //create an array of double on gpu und host
            gBins = new GridBuffer<double, DIM1 > (DataSpace<DIM1 > (realNumBins));
            binReduced = new double[realNumBins];
            for (int i = 0; i < realNumBins; ++i)
            {
                binReduced[i] = 0.0;
            }

            if (writeToFile)
            {
                outFile.open(filename.c_str(), std::ofstream::out | std::ostream::trunc);
                if (!outFile)
                {
                    std::cerr << "Can't open file [" << filename << "] for output, diasble analyser output. " << std::endl;
                    writeToFile = false;
                }
                //create header of the file
                outFile << "#step <" << minEnergy << " ";
                float_X binEnergy = (maxEnergy - minEnergy) * UNITCONV_Joule_to_keV * UNIT_ENERGY / (float) numBins;
                for (int i = 1; i < realNumBins - 1; ++i)
                {

                    outFile << minEnergy + ((float) i * binEnergy) << " ";
                }
                outFile << ">" << maxEnergy << " count" << std::endl;
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

            __delete(gBins);
            delete[] binReduced;
        }
    }

    template< uint32_t AREA>
    void calBinEnergyParticles(uint32_t currentStep)
    {
        gBins->getDeviceBuffer().setValue(0);
        dim3 block(MappingDesc::SuperCellSize::getDataSpace());

        /** Assumption: distanceToDetector >> simulated Area in y-Direction
         *          AND     simulated area in X,Z << slit  */
        double maximumSlopeToDetectorX = 0.0; /*0.0 is disabled detector*/
        double maximumSlopeToDetectorZ = 0.0; /*0.0 is disabled detector*/
        if (enableDetector)
        {
            maximumSlopeToDetectorX = (slitDetectorX / 2.0) / (distanceToDetector);
            maximumSlopeToDetectorZ = (slitDetectorZ / 2.0) / (distanceToDetector);
            //maximumSlopeToDetector = (radiusDetector * radiusDetector) / (distanceToDetector * distanceToDetector);
        }

        __picKernelArea(kernelBinEnergyParticles, *cellDescription, AREA)
            (block, (realNumBins) * sizeof (float_X))
            (particles->getDeviceParticlesBox(),
             gBins->getDeviceBuffer().getDataBox(), numBins, minEnergy,
             maxEnergy, maximumSlopeToDetectorX, maximumSlopeToDetectorZ);

        gBins->deviceToHost();

        reduce(nvidia::functors::Add(),
               binReduced,
               gBins->getHostBuffer().getBasePointer(),
               realNumBins, mpi::reduceMethods::Reduce());


        if (writeToFile)
        {
            typedef std::numeric_limits< float_64 > dbl;

            outFile.precision(dbl::digits10);

            //write data to file
            double count_particles = 0.0;
            outFile << currentStep << " "
                << std::scientific; // for floating points, ignored for ints

            for (int i = 0; i < realNumBins; ++i)
            {
                count_particles += double( binReduced[i]);
                outFile << std::scientific << (binReduced[i]) * double(NUM_EL_PER_PARTICLE) << " ";
            }
            outFile << std::scientific << count_particles * double(NUM_EL_PER_PARTICLE)
                << std::endl;
            // endl: Flush any step to the file.
            // Thus, we will have data if the program should crash.
        }
    }

};

}

#endif	/* BINENERGYPARTICLES_HPP */

