/**
 * Copyright 2013-2016 Axel Huebl, Felix Schmitt, Heiko Burau,
 *                     Rene Widera, Richard Pausch, Benjamin Worpitz
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
#include <fstream>
#include <mpi.h>

#include "pmacc_types.hpp"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include "simulation_classTypes.hpp"
#include "mappings/kernel/AreaMapping.hpp"
#include "plugins/ISimulationPlugin.hpp"

#include "mpi/reduceMethods/Reduce.hpp"
#include "mpi/MPIReduce.hpp"
#include "nvidia/functors/Add.hpp"

#include "algorithms/Gamma.hpp"

#include "common/txtFileHandling.hpp"

namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;

/** This kernel computes the kinetic and total energy summed over
 *  all particles of a species.
 **/
template<class ParBox, class DBox, class Mapping>
__global__ void kernelEnergyParticles(ParBox pb,
                                      DBox gEnergy,
                                      Mapping mapper)
{

    typedef typename ParBox::FramePtr FramePtr;
    __shared__ typename PMacc::traits::GetEmptyDefaultConstructibleType<FramePtr>::type frame; /* pointer to particle data frame */
    __shared__ float_X shEnergyKin; /* shared kinetic energy */
    __shared__ float_X shEnergy; /* shared total energy */

    float_X _local_energyKin = float_X(0.0); /* sum kinetic energy for this thread */
    float_X _local_energy = float_X(0.0); /* sum total energy for this thread */


    typedef typename Mapping::SuperCellSize SuperCellSize;

    const DataSpace<simDim > threadIndex(threadIdx);
    const int linearThreadIdx = DataSpaceOperations<simDim>::template map<SuperCellSize > (threadIndex);

    if (linearThreadIdx == 0) /* only thread 0 runs initial set up */
    {
        const DataSpace<simDim> superCellIdx(mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx)));
        frame = pb.getLastFrame(superCellIdx); /* get first(=last) frame */
        shEnergyKin = float_X(0.0); /* set shared kinetic energy to zero */
        shEnergy = float_X(0.0); /* set shared total energy to zero */
    }

    __syncthreads(); /* wait till thread 0 finishes set up */
    if (!frame.isValid())
        return; /* end kernel if we have no frames */

    /* this checks if the data loaded by a thread is filled with a particle
     * or not. Only applies to the first loaded frame (=last frame) */
    /* BUGFIX to issue #538
     * volatile prohibits that the compiler creates wrong code*/
    volatile bool isParticle = frame[linearThreadIdx][multiMask_];

    while (frame.isValid())
    {
        if (isParticle)
        {

            PMACC_AUTO(particle,frame[linearThreadIdx]); /* get one particle */
            const float3_X mom = particle[momentum_]; /* get particle momentum */
            /* and compute square of absolute momentum of one particle: */
            const float_X mom2 = mom.x() * mom.x() + mom.y() * mom.y() + mom.z() * mom.z();

            const float_X weighting = particle[weighting_]; /* get macro particle weighting */
            const float_X mass = attribute::getMass(weighting,particle); /* compute mass using weighting */
            const float_X c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

            Gamma<> calcGamma; /* functor for computing relativistic gamma factor */
            const float_X gamma = calcGamma(mom, mass); /* compute relativistic gamma */

            if (gamma < GAMMA_THRESH) /* if particle energy is low enough: */
            {
                /* not relativistic: use equation with more precision */
                _local_energyKin += mom2 / (2.0f * mass);
            }
            else /* if particle is relativistic */
            {
                /* kinetic energy for particles: E = (gamma - 1) * m * c^2
                 *                                    gamma = sqrt( 1 + (p/m/c)^2 )
                 * _local_energyKin += (algorithms::math::sqrt(mom2 / (mass * mass * c2) 
                 *                                             + 1.) - 1.) * mass * c2;
                 */
                _local_energyKin += (gamma - float_X(1.0)) * mass*c2;
            }

            /* total energy for particles: E^2 = p^2*c^2 + m^2*c^4
             *                                   = c^2 * [p^2 + m^2*c^2]
             */
            _local_energy += algorithms::math::sqrt(mom2 + mass * mass * c2) * SPEED_OF_LIGHT;

        }
        __syncthreads(); /* wait till all threads have added their particle energies */

        /* get next particle frame */
        if (linearThreadIdx == 0)
        {
            /* set frame to next particle frame */
            frame = pb.getPreviousFrame(frame);
        }
        isParticle = true; /* all following frames are filled with particles */
        __syncthreads(); /* wait till thread 0 is done */
    }

    /* add energies on block level using shared memory */
    atomicAddWrapper(&shEnergyKin, _local_energyKin); /* add kinetic energy */
    atomicAddWrapper(&shEnergy, _local_energy);       /* add total energy */

    __syncthreads(); /* wait till all threads have added their energies */

    /* add energies on global level using global memory */
    if (linearThreadIdx == 0) /* only done by thread 0 of a block */
    {
        atomicAddWrapper(&(gEnergy[0]), (float_64) (shEnergyKin)); /* add kinetic energy */
        atomicAddWrapper(&(gEnergy[1]), (float_64) (shEnergy));    /* add total energy */
    }
}

template<class ParticlesType>
class EnergyParticles : public ISimulationPlugin
{
private:
    typedef MappingDesc::SuperCellSize SuperCellSize;

    ParticlesType *particles; /* pointer to particle data */

    GridBuffer<float_64, DIM1> *gEnergy; /* energy values (global on GPU) */
    MappingDesc *cellDescription;
    uint32_t notifyFrequency; /* periodocity of computing the particle energy */

    std::string analyzerName; /* name (used for output file too) */
    std::string analyzerPrefix; /* prefix used for command line arguments */
    std::string filename; /* output file name */

    std::ofstream outFile; /* file output stream */
    bool writeToFile;   /* only rank 0 creates a file */

    mpi::MPIReduce reduce; /* MPI reduce to add all energies over several GPUs */

public:

    EnergyParticles() :
    analyzerName("EnergyParticles: calculate the energy of a species"),
    analyzerPrefix(ParticlesType::FrameType::getName() + std::string("_energy")),
    filename(analyzerPrefix + ".dat"),
    particles(NULL),
    gEnergy(NULL),
    cellDescription(NULL),
    notifyFrequency(0),
    writeToFile(false)
    {
        /* register this plugin */
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~EnergyParticles()
    {

    }

  /** this code is executed if the current time step is supposed to compute
   * the energy **/
    void notify(uint32_t currentStep)
    {
        DataConnector &dc = Environment<>::get().DataConnector(); /* get data connector */

        /* use data connector to get particle data */
        particles = &(dc.getData<ParticlesType > (ParticlesType::FrameType::getName(), true));

        /* call the method that calls the plugin kernel */
        calculateEnergyParticles < CORE + BORDER > (currentStep);
    }

  /** method used by plugin controller to get --help description **/
    void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ((analyzerPrefix + ".period").c_str(),
             po::value<uint32_t > (&notifyFrequency),
             "compute kinetic and total energy [for each n-th step] enable analyser by setting a non-zero value");
    }

  /** method giving the plugin name (used by plugin control) **/
    std::string pluginGetName() const
    {
        return analyzerName;
    }

  /** set cell description in this plugin **/
    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

private:

    /** method to initialize plugin output and variables **/
    void pluginLoad()
    {
        if (notifyFrequency > 0) /* only if plugin is called at least once */
        {
            /* decide which MPI-rank writes output: */
            writeToFile = reduce.hasResult(mpi::reduceMethods::Reduce());

            /* create two ints on gpu and host: */
            gEnergy = new GridBuffer<float_64, DIM1 > (DataSpace<DIM1 > (2));

            if (writeToFile) /* only MPI rank that writes to file: */
            {
                /* open output file */
                outFile.open(filename.c_str(), std::ofstream::out | std::ostream::trunc);

                /* error handling: */
                if (!outFile)
                {
                    std::cerr << "Can't open file [" << filename
                              << "] for output, diasble analyser output. " << std::endl;
                    writeToFile = false;
                }

                /* create header of the file */
                outFile << "#step Ekin_Joule E_Joule" << " \n";
            }

            /* set how often the plugin should be executed while PIConGPU is running */
            Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyFrequency);
        }
    }

    /** method to quit plugin **/
    void pluginUnload()
    {
        if (notifyFrequency > 0) /* only if plugin is called at least once */
        {
            if (writeToFile)
            {
                outFile.flush();
                outFile << std::endl; /* now all data is written to file */

                /* error handling: */
                if (outFile.fail())
                    std::cerr << "Error on flushing file [" << filename << "]. " << std::endl;
                outFile.close();
            }

            __delete(gEnergy); /* free global memory on GPU */
        }
    }

    void restart(uint32_t restartStep, const std::string restartDirectory)
    {
        if( !writeToFile )
            return;

        writeToFile = restoreTxtFile( outFile,
                                      filename,
                                      restartStep,
                                      restartDirectory );
    }

    void checkpoint(uint32_t currentStep, const std::string checkpointDirectory)
    {
        if( !writeToFile )
            return;

        checkpointTxtFile( outFile,
                           filename,
                           currentStep,
                           checkpointDirectory );
    }

    /** method to call analysis and plugin-kernel calls **/
    template< uint32_t AREA>
    void calculateEnergyParticles(uint32_t currentStep)
    {
        gEnergy->getDeviceBuffer().setValue(0.0); /* init global energy with zero */
        dim3 block(MappingDesc::SuperCellSize::toRT().toDim3()); /* GPU parallelization */

        /* kernel call = sum all particle energies on GPU */
        __picKernelArea(kernelEnergyParticles, *cellDescription, AREA)
            (block)
            (particles->getDeviceParticlesBox(),
             gEnergy->getDeviceBuffer().getDataBox());

        gEnergy->deviceToHost(); /* get energy from GPU */

        float_64 reducedEnergy[2]; /* create storage for kinetic and total energy */

        /* add energies from all GPUs using MPI: */
        reduce(nvidia::functors::Add(),
               reducedEnergy,
               gEnergy->getHostBuffer().getBasePointer(),
               2,
               mpi::reduceMethods::Reduce());

        /* print timestep, kinetic energy and total energy to file: */
        if (writeToFile)
        {
            typedef std::numeric_limits< float_64 > dbl;

            outFile.precision(dbl::digits10);
            outFile << currentStep << " "
                    << std::scientific
                    << reducedEnergy[0] * UNIT_ENERGY << " "
                    << reducedEnergy[1] * UNIT_ENERGY << std::endl;
        }
    }

};

}

