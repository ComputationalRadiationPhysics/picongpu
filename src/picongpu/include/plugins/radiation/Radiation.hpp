/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch, Klaus Steiniger,
 * Felix Schmitt
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
#include <stdlib.h>

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"
#include "basicOperations.hpp"
#include "dimensions/DataSpaceOperations.hpp"

#include "simulation_classTypes.hpp"
#include "mappings/kernel/AreaMapping.hpp"
#include "plugins/ISimulationPlugin.hpp"


#include "mpi/reduceMethods/Reduce.hpp"
#include "mpi/MPIReduce.hpp"
#include "nvidia/functors/Add.hpp"

#include "sys/stat.h"

#include "plugins/radiation/Radiation.kernel"

/* libSpash data output */
#include <splash/splash.h>

namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;



///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////  Radiation Analyzer Class  ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template<class ParticlesType>
class Radiation : public ISimulationPlugin
{
private:

    typedef MappingDesc::SuperCellSize SuperCellSize;

    typedef PIConGPUVerboseRadiation radLog;

    /**
     * At the moment the ParticlesType is PIC_ELECTRONS
     * (This special class which stores information about the momentum
     * of the last two time steps)
     */
    ParticlesType *particles;

    /**
     * Object that stores the complex radiated amplitude on host and device.
     * Radiated amplitude is a function of theta (looking direction) and
     * frequency. Layout of the radiation array is:
     * [omega_1(theta_1),omega_2(theta_1),...,omega_N-omega(theta_1),
     *   omega_1(theta_2),omega_2(theta_2),...,omega_N-omega(theta_N-theta)]
     */
    GridBuffer<Amplitude, DIM1> *radiation;
    radiation_frequencies::InitFreqFunctor freqInit;
    radiation_frequencies::FreqFunctor freqFkt;

    MappingDesc *cellDescription;
    uint32_t notifyFrequency;
    uint32_t dumpPeriod;
    uint32_t radStart;
    uint32_t radEnd;

    std::string analyzerName;
    std::string analyzerPrefix;
    std::string filename_prefix;
    bool totalRad;
    bool lastRad;
    std::string folderLastRad;
    std::string folderTotalRad;
    std::string pathOmegaList;
    bool radPerGPU;
    std::string folderRadPerGPU;
    DataSpace<simDim> lastGPUpos;

    /**
     * Data structure for storage and summation of the intermediate values of
     * the calculated Amplitude from every host for every direction and
     * frequency.
     */
    Amplitude* timeSumArray;

    bool isMaster;

    uint32_t currentStep;
    uint32_t lastStep;

    bool radRestart;
    std::string pathRestart;

    mpi::MPIReduce reduce;

public:

    Radiation(std::string name, std::string prefix) :
    analyzerName(name),
    analyzerPrefix(prefix),
    filename_prefix(name),
    particles(NULL),
    radiation(NULL),
    cellDescription(NULL),
    notifyFrequency(0),
    dumpPeriod(0),
    totalRad(false),
    lastRad(false),
    timeSumArray(NULL),
    isMaster(false),
    currentStep(0),
    radPerGPU(false),
    lastStep(0),
    radRestart(false)
    {

        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~Radiation()
    {

    }

    /**
     * This function represents what is actually calculated if the analyzer
     * is called. Here, one only sets the particles pointer to the data of
     * the latest time step and calls the 'calculateRadiationParticles'
     * function if for the actual time step radiation is to be calculated.
     * @param currentStep
     */
    void notify(uint32_t currentStep)
    {

        DataConnector &dc = Environment<>::get().DataConnector();

        particles = &(dc.getData<ParticlesType > (ParticlesType::FrameType::getName(), true));

        if (currentStep >= radStart)
        {
            // radEnd = 0 is default, calculates radiation until simulation
            // end
            if (currentStep <= radEnd || radEnd == 0)
            {
                log<radLog::SIMULATION_STATE > ("radiation gets calculated: timestep %1% ") % currentStep;

                /* CORE + BORDER is PIC black magic, currently not needed
                 *
                 */
                calculateRadiationParticles < CORE + BORDER > (currentStep);

                log<radLog::SIMULATION_STATE > ("radiation got calculated: timestep %1% ") % currentStep;
            }
        }
    }

    void pluginRegisterHelp(po::options_description& desc)
    {

        desc.add_options()
            ((analyzerPrefix + ".period").c_str(), po::value<uint32_t > (&notifyFrequency), "enable analyser [for each n-th step]")
            ((analyzerPrefix + ".dump").c_str(), po::value<uint32_t > (&dumpPeriod)->default_value(0), "dump integrated radiation from last dumped step [for each n-th step] (0 = only print data at end of simulation)")
            ((analyzerPrefix + ".lastRadiation").c_str(), po::value<bool > (&lastRad)->default_value(false), "enable(1)/disable(0) calculation integrated radiation from last dumped step")
            ((analyzerPrefix + ".folderLastRad").c_str(), po::value<std::string > (&folderLastRad)->default_value("lastRad"), "folder in which the integrated radiation from last dumped step is written")
            ((analyzerPrefix + ".totalRadiation").c_str(), po::value<bool > (&totalRad)->default_value(false), "enable(1)/disable(0) calculation integrated radiation from start of simulation")
            ((analyzerPrefix + ".folderTotalRad").c_str(), po::value<std::string > (&folderTotalRad)->default_value("totalRad"), "folder in which the integrated radiation from start of simulation is written")
            ((analyzerPrefix + ".start").c_str(), po::value<uint32_t > (&radStart)->default_value(2), "time index when radiation should start with calculation")
            ((analyzerPrefix + ".end").c_str(), po::value<uint32_t > (&radEnd)->default_value(0), "time index when radiation should end with calculation")
            ((analyzerPrefix + ".omegaList").c_str(), po::value<std::string > (&pathOmegaList)->default_value("_noPath_"), "path to file containing all frequencies to calculate")
            ((analyzerPrefix + ".radPerGPU").c_str(), po::value<bool > (&radPerGPU)->default_value(false), "enable(1)/disable(0) radiation output from each GPU individually")
            ((analyzerPrefix + ".folderRadPerGPU").c_str(), po::value<std::string > (&folderRadPerGPU)->default_value("radPerGPU"), "folder in which the radiation of each GPU is written")
            ((analyzerPrefix + ".restart").c_str(), po::value<bool > (&radRestart)->default_value(false), "enable(1)/disable(0) restart flag");
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

    /**
     * The plugin is loaded on every host pc, and therefor this function is
     * executed on every host pc.
     * One host with MPI rank 0 is defined to be the master.
     * It creates a folder where all the
     * results are saved and, depending on the type of radiation calculation,
     * creates an additional data structure for the summation of all
     * intermediate values.
     * On every host data structure for storage of the calculated radiation
     * is created.       */
    void pluginLoad()
    {
        if (notifyFrequency > 0)
        {
            /*only rank 0 create a file*/
            isMaster = reduce.hasResult(mpi::reduceMethods::Reduce());
            const int elements_amplitude = radiation_frequencies::N_omega * parameters::N_observer; // storage for amplitude results on GPU

            radiation = new GridBuffer<Amplitude, DIM1 > (DataSpace<DIM1 > (elements_amplitude)); //create one int on gpu und host

            freqInit.Init(pathOmegaList);
            freqFkt = freqInit.getFunctor();


            Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyFrequency);
            PMacc::Filesystem<simDim>& fs = Environment<simDim>::get().Filesystem();

            if (isMaster && totalRad)
            {
                fs.createDirectory("radRestart");
                fs.setDirectoryPermissions("radRestart");
            }


            if (isMaster && radPerGPU)
            {
                fs.createDirectory(folderRadPerGPU);
                fs.setDirectoryPermissions(folderRadPerGPU);
            }

            if (isMaster && totalRad)
            {
                //create folder for total output
                fs.createDirectory(folderTotalRad);
                fs.setDirectoryPermissions(folderTotalRad);
                timeSumArray = new Amplitude[elements_amplitude];
                for (int i = 0; i < elements_amplitude; ++i)
                    timeSumArray[i] = Amplitude();
            }
            if (isMaster && lastRad)
            {
                //create folder for total output
                fs.createDirectory(folderLastRad);
                fs.setDirectoryPermissions(folderLastRad);
            }

        }
    }

    void pluginUnload()
    {
        if (notifyFrequency > 0)
        {

            // Some funny things that make it possible for the kernel to calculate
            // the absolut position of the particles
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            DataSpace<simDim> localSize(subGrid.getLocalDomain().size);
            const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
            DataSpace<simDim> globalOffset(subGrid.getLocalDomain().offset);
            globalOffset.y() += (localSize.y() * numSlides);

            //only print data at end of simulation
            if (dumpPeriod == 0)
                combineData(globalOffset);
            if (isMaster && totalRad)
            {
                __deleteArray(timeSumArray);
            }

            __delete(radiation);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    /**
     * This function is called by the calculateRadiationParticles() function
     * if storing of intermediate results is activated (dumpPeriod != 0)
     * otherwise it is invoked by pluginUnload().
     *
     * On every host the calculated radiation (radiation from the particles
     * on that device for all directions and frequencies) is transferred
     * from the gpu to the host, and then the data from all hosts is
     * combined on the master host. Hence, the emitted radiation of all
     * particles for every direction and step is available on the master
     * host.
     */
    void combineData(const DataSpace<simDim> currentGPUpos)
    {

        const unsigned int elements_amplitude = radiation_frequencies::N_omega * parameters::N_observer; // storage for amplitude results on GPU
        Amplitude *result = new Amplitude[elements_amplitude];


        radiation->deviceToHost();
        __getTransactionEvent().waitForFinished();

        if (radPerGPU)
        {
            if (lastGPUpos == currentGPUpos)
            {
                std::stringstream last_time_step_str;
                std::stringstream current_time_step_str;
                std::stringstream GPUposX;
                std::stringstream GPUposY;
                std::stringstream GPUposZ;


                last_time_step_str << lastStep;
                current_time_step_str << currentStep;
                GPUposX << currentGPUpos.x();
                GPUposY << currentGPUpos.y();
                // if simDim==DIM2 no z-component is defined -> set it to zero
                #if(SIMDIM == DIM3)
                  GPUposZ << currentGPUpos.z();
                #else
                  GPUposZ << 0;
                #endif

                writeFile(radiation->getHostBuffer().getBasePointer(), folderRadPerGPU + "/" + filename_prefix
                          + "_radPerGPU_pos_" + GPUposX.str() + "_" + GPUposY.str() + "_" + GPUposZ.str()
                          + "_time_" + last_time_step_str.str() + "-" + current_time_step_str.str() + ".dat");
            }
        }

        reduce(nvidia::functors::Add(),
               result,
               radiation->getHostBuffer().getBasePointer(),
               elements_amplitude,
               mpi::reduceMethods::Reduce()
               );

        if (isMaster)
        {
            std::stringstream o_step;
            o_step << currentStep;
            if (totalRad)
            {
                /*
                 * totalRad writes to file the total emitted radiation up
                 * to the current time step.
                 */

                if (radRestart)
                {
                    std::stringstream o_bu_step;
                    o_bu_step << currentStep - dumpPeriod;

                    loadBackup(timeSumArray, std::string("radRestart") + "/" + std::string("radRestart") + "_" + o_bu_step.str() + ".dat");
                    radRestart = false; // reset restart flag
                }


                for (unsigned int i = 0; i < elements_amplitude; ++i)
                    timeSumArray[i] += result[i];
                writeFile(timeSumArray, folderTotalRad + "/" + filename_prefix + "_" + o_step.str() + ".dat");
                writeBackup(timeSumArray, std::string("radRestart") + "/" + std::string("radRestart") + "_" + o_step.str() + ".dat");

                writeHDF5file(timeSumArray, std::string("dummy"));
            }

            if (lastRad)
                /*
                 * lastRad writes to file the emitted radiation only from
                 * the current time step. That is, radiation from previous
                 * time steps is neglected.
                 */
                writeFile(result, folderLastRad + "/" + filename_prefix + "_" + o_step.str() + ".dat");
        }

        __deleteArray(result);

        lastStep = currentStep;
        lastGPUpos = currentGPUpos;
    }

    /**
     * From the collected data from all hosts the radiated intensity is
     * calculated by calculating the absolute value squared and multiplying
     * this with with the appropriate physics constants.
     * @param values
     * @param name
     */
    void writeFile(Amplitude* values, std::string name)
    {
        std::ofstream outFile;
        outFile.open(name.c_str(), std::ofstream::out | std::ostream::trunc);
        if (!outFile)
        {
            std::cerr << "Can't open file [" << name << "] for output, diasble analyser output. " << std::endl;
            isMaster = false; // no Master anymore -> no process is able to write
        }
        else
        {
            for (unsigned int index_direction = 0; index_direction < parameters::N_observer; ++index_direction) // over all directions
            {
                for (unsigned index_omega = 0; index_omega < radiation_frequencies::N_omega; ++index_omega) // over all frequencies
                {
                    // Take Amplitude for one direction and frequency,
                    // calculate the square of the absolute value
                    // and write to file.
                    outFile <<
                        values[index_omega + index_direction * radiation_frequencies::N_omega].calc_radiation() * UNIT_ENERGY * UNIT_TIME << "\t";

                }
                outFile << std::endl;
            }
            outFile.flush();
            outFile << std::endl; //now all data are written to file

            if (outFile.fail())
                std::cerr << "Error on flushing file [" << name << "]. " << std::endl;
            outFile.close();
        }
    }

public:
    void restart(uint32_t timeStep, const std::string restartDirectory)
    {

    }

    void checkpoint(uint32_t timeStep, const std::string restartDirectory)
    {
      std::stringstream t_step;
      t_step << timeStep;

      writeBackup(timeSumArray, restartDirectory + "/" + std::string("radRestart") + "_" + t_step.str() + ".dat");

    }


private:

    void writeHDF5file(Amplitude* values, std::string name)
    {
        splash::SerialDataCollector HDF5dataFile(1);
        splash::DataCollector::FileCreationAttr fAttr;

        splash::DataCollector::initFileCreationAttr(fAttr);

        //        double a = 9.876;
        //        double vec[5];
        //        vec[1] = 123.45;


        const double six = sizeof(Amplitude)/sizeof(double);

        std::ostringstream filename;
        filename << "radHDF5_" << currentStep << ".h5";
        
        HDF5dataFile.open(filename.str().c_str(), fAttr);

        typename PICToSplash<double>::type radSplashType;

        HDF5dataFile.write(currentStep,
                           radSplashType,
                           3,
                           splash::Selection(splash::Dimensions(six,
                                                                radiation_frequencies::N_omega,
                                                                parameters::N_observer)),
                           "amplitudes",
                           values);

        //HDF5dataFile.writeAttribute(currentStep,
        //                            radSplashType,
        //                            "dataName",
        //                            "attrName",
        //                            &a);

        HDF5dataFile.close();
        std::cout << "rad_output to HDF5" << std::endl;

    }

    void writeBackup(Amplitude* values, std::string name)
    {

        std::ofstream outFile;
        outFile.open(name.c_str(), std::ofstream::out | std::ofstream::binary);
        if (!outFile)
        {
            std::cerr << "Can't open file [" << name << "] for backup, diasble analyser output. " << std::endl;
            isMaster = false; // no Master anymore -> no process is able to write
        }
        else
        {
            outFile.write((char*) values, sizeof (Amplitude) * parameters::N_observer * radiation_frequencies::N_omega);
        }

        outFile.close();
    }

    void loadBackup(Amplitude* values, std::string name)
    {

        std::ifstream inFile;
        inFile.open(name.c_str(), std::ifstream::in | std::ifstream::binary);
        if (!inFile)
        {
            std::cerr << "Can't open file [" << name << "] for loading backup. " << std::endl;
        }
        else
        {
            inFile.read((char*) values, sizeof (Amplitude) * parameters::N_observer * radiation_frequencies::N_omega);
            std::cout << "Radiation: backup files have been loaded." << std::endl;
        }

        inFile.close();
    }

    /**
     * This functions calls the radiation kernel. It specifies how the
     * calculation is parallelized.
     *      gridDim_rad is the number of Thread-Blocks in a grid
     *      blockDim_rad is the number of threads per block
     *
     * -----------------------------------------------------------
     * | Grid                                                    |
     * |   --------------   --------------                       |
     * |   |   Block 0  |   |   Block 1  |                       |
     * |   |o      o    |   |o      o    |                       |
     * |   |o      o    |   |o      o    |                       |
     * |   |th1    th2  |   |th1    th2  |                       |
     * |   --------------   --------------                       |
     * -----------------------------------------------------------
     *
     * !!! The TEMPLATE parameter is not used anymore.
     * !!! But the calculations it is supposed to do is hard coded in the
     *     kernel.
     * !!! THIS NEEDS TO BE CHANGED !!!
     *
     * @param currentStep
     */
    template< uint32_t AREA> /*This Template Parameter is not used anymore*/
    void calculateRadiationParticles(uint32_t currentStep)
    {
        this->currentStep = currentStep;

        /* the parallelization is ONLY over directions:
         * (a combinded parallelization over direction AND frequencies
         *  turned out to be slower on fermis (couple percent) and
         *  definitly slower on kepler k20)
         */
        const int N_observer = parameters::N_observer;
        const dim3 gridDim_rad(N_observer);

        /* number of threads per block = number of cells in a super cell
         *          = number of particles in a Frame
         *          (THIS IS PIConGPU SPECIFIC)
         * A Frame is the entity that stores particles.
         * A super cell can have many Frames.
         * Particles in a Frame can be accessed in parallel.
         */

        const dim3 blockDim_rad(PMacc::math::CT::volume<typename MappingDesc::SuperCellSize>::type::value);


        // std::cout<<"Grid: "<<gridDim_rad.x()<<" "<<gridDim_rad.y()<<" "<<gridDim_rad.z()<<std::endl;
        // std::cout<<"Block: "<<blockDim_rad.x()<<" "<<blockDim_rad.y()<<" "<<blockDim_rad.z()<<std::endl;


        // Some funny things that make it possible for the kernel to calculate
        // the absolut position of the particles
        DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
        const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        DataSpace<simDim> globalOffset(subGrid.getLocalDomain().offset);
        globalOffset.y() += (localSize.y() * numSlides);


        // PIC-like kernel call of the radiation kernel
        __cudaKernel(kernelRadiationParticles)
            (gridDim_rad, blockDim_rad)
            (
             /*Pointer to particles memory on the device*/
             particles->getDeviceParticlesBox(),

             /*Pointer to memory of radiated amplitude on the device*/
             radiation->getDeviceBuffer().getDataBox(),
             globalOffset,
             currentStep, *cellDescription,
	     freqFkt,
	     subGrid.getGlobalDomain().size
             );

        if (dumpPeriod != 0 && currentStep % dumpPeriod == 0)
        {
            combineData(globalOffset);
            radiation->getDeviceBuffer().reset(false);
        }

    }

};

}



