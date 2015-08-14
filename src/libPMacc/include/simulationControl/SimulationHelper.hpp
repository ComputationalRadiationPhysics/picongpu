/**
 * Copyright 2013-2015 Axel Huebl, Felix Schmitt, Rene Widera, Alexander Debus,
 *                     Benjamin Worpitz, Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "types.h"

#include "mappings/simulation/GridController.hpp"
#include "dimensions/DataSpace.hpp"
#include "TimeInterval.hpp"

#include "dataManagement/DataConnector.hpp"


#include "pluginSystem/IPlugin.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>

namespace PMacc
{

/**
 * Abstract base class for simulations.
 *
 * Use this helper class to write your own concrete simulations
 * by binding pure virtual methods.
 *
 * @tparam DIM base dimension for the simulation (2-3)
 */
template<unsigned DIM>
class SimulationHelper : public IPlugin
{
public:

    /**
     * Constructor
     *
     */
    SimulationHelper() :
    runSteps(0),
    checkpointPeriod(0),
    checkpointDirectory("checkpoints"),
    numCheckpoints(0),
    restartStep(-1),
    restartDirectory("checkpoints"),
    restartRequested(false),
    CHECKPOINT_MASTER_FILE("checkpoints.txt")
    {
        tSimulation.toggleStart();
        tInit.toggleStart();

    }

    virtual ~SimulationHelper()
    {
        tSimulation.toggleEnd();
        if (output)
        {
            std::cout << "full simulation time: " <<
                tSimulation.printInterval() << " = " <<
                (uint64_t) (tSimulation.getInterval() / 1000.) << " sec" << std::endl;
        }
    }

    /**
     * Must describe one iteration (step).
     *
     * This function is called automatically.
     */
    virtual void runOneStep(uint32_t currentStep) = 0;

    /**
     * Initializes simulation state.
     *
     * @return returns the first step of the simulation
     */
    virtual uint32_t init() = 0;


    /**
     * Check if moving window work must do
     *
     * If no moving window is needed the implementation of this function can be empty
     *
     * @param currentStep simulation step
     */
    virtual void movingWindowCheck(uint32_t currentStep) = 0;

    /**
     * Notifies registered output classes.
     *
     * This function is called automatically.
     *
     *  @param currentStep simulation step
     */
    virtual void dumpOneStep(uint32_t currentStep)
    {
        Environment<DIM>::get().DataConnector().invalidate();

        /* trigger notification */
        Environment<DIM>::get().PluginConnector().notifyPlugins(currentStep);

        /* trigger checkpoint notification */
        if (checkpointPeriod && (currentStep % checkpointPeriod == 0))
        {
            /* first synchronize: if something failed, we can spare the time
             * for the checkpoint writing */
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            GridController<DIM> &gc = Environment<DIM>::get().GridController();
            /* can be spared for better scalings, but allows to spare the
             * time for checkpointing if some ranks died */
            MPI_CHECK(MPI_Barrier(gc.getCommunicator().getMPIComm()));

            /* create directory containing checkpoints  */
            if (numCheckpoints == 0)
            {
                Environment<DIM>::get().Filesystem().createDirectoryWithPermissions(checkpointDirectory);
            }

            Environment<DIM>::get().PluginConnector().checkpointPlugins(currentStep,
                                                                        checkpointDirectory);

            /* important synchronize: only if no errors occured until this
             * point guarantees that a checkpoint is usable */
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            /* \todo in an ideal world with MPI-3, this would be an
             * MPI_Ibarrier call and this function would return a MPI_Request
             * that could be checked */
            MPI_CHECK(MPI_Barrier(gc.getCommunicator().getMPIComm()));

            if (gc.getGlobalRank() == 0)
            {
                writeCheckpointStep(currentStep);
            }
            numCheckpoints++;
        }
    }

    GridController<DIM> & getGridController()
    {
        return Environment<DIM>::get().GridController();
    }

    void dumpTimes(TimeIntervall &tSimCalculation, TimeIntervall&, double& roundAvg, uint32_t currentStep)
    {
        /*dump 100% after simulation*/
        if (output && progress && (currentStep % showProgressAnyStep) == 0)
        {
            tSimCalculation.toggleEnd();
            std::cout << std::setw(3) <<
                (uint16_t) ((double) currentStep / (double) runSteps * 100.) <<
                " % = " << std::setw(8) << currentStep <<
                " | time elapsed:" <<
                std::setw(25) << tSimCalculation.printInterval() << " | avg time per step: " <<
                TimeIntervall::printeTime(roundAvg / (double) showProgressAnyStep) << std::endl;
            std::cout.flush();

            roundAvg = 0.0; //clear round avg timer
        }

    }

    /**
     * Begin the simulation.
     */
    void startSimulation()
    {
        uint32_t currentStep = init();
        tInit.toggleEnd();
        if (output)
        {
            std::cout << "initialization time: " << tInit.printInterval() <<
                " = " <<
                (int) (tInit.getInterval() / 1000.) << " sec" << std::endl;
        }

        TimeIntervall tSimCalculation;
        TimeIntervall tRound;
        double roundAvg = 0.0;

    /* dump initial step if simulation starts without restart */
    if (currentStep == 0)
    {
        /* Since in the main loop movingWindow is called always before the dump, we also call it here for consistency.
        This becomes only important, if movingWindowCheck does more than merely checking for a slide.
        TO DO in a new feature: Turn this into a general hook for pre-checks (window slides are just one possible action). */
        movingWindowCheck(currentStep);
        dumpOneStep(currentStep);
    }
    else
    {
        currentStep--; //We dump before calculation, thus we must go one step back when doing a restart.
        movingWindowCheck(currentStep); //If we restart at any step check if we must slide.
    }

        /* dump 0% output */
        dumpTimes(tSimCalculation, tRound, roundAvg, currentStep);
        while (currentStep < runSteps)
        {
            tRound.toggleStart();
            runOneStep(currentStep);
            tRound.toggleEnd();
            roundAvg += tRound.getInterval();

            currentStep++;
            /*output after a round*/
            dumpTimes(tSimCalculation, tRound, roundAvg, currentStep);

            movingWindowCheck(currentStep);
            /*dump after simulated step*/
            dumpOneStep(currentStep);
        }

        //simulatation end
        Environment<>::get().Manager().waitForAllTasks();

        tSimCalculation.toggleEnd();

        if (output)
        {
            std::cout << "calculation  simulation time: " <<
                tSimCalculation.printInterval() << " = " <<
                (int) (tSimCalculation.getInterval() / 1000.) << " sec" << std::endl;
        }

    }

    virtual void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ("steps,s", po::value<uint32_t > (&runSteps), "Simulation steps")
            ("percent,p", po::value<uint16_t > (&progress)->default_value(5),
             "Print time statistics after p percent to stdout")
            ("restart", po::value<bool>(&restartRequested)->zero_tokens(), "Restart simulation")
            ("restart-directory", po::value<std::string>(&restartDirectory)->default_value(restartDirectory),
             "Directory containing checkpoints for a restart")
            ("restart-step", po::value<int32_t>(&restartStep), "Checkpoint step to restart from")
            ("checkpoints", po::value<uint32_t>(&checkpointPeriod), "Period for checkpoint creation")
            ("checkpoint-directory", po::value<std::string>(&checkpointDirectory)->default_value(checkpointDirectory),
             "Directory for checkpoints");
    }

    std::string pluginGetName() const
    {
        return "SimulationHelper";
    }

    void pluginLoad()
    {
        calcProgress();

        output = (getGridController().getGlobalRank() == 0);
    }

    void pluginUnload()
    {
    }

    void restart(uint32_t, const std::string)
    {
    }

    void checkpoint(uint32_t, const std::string)
    {
    }

protected:
    /* number of simulation steps to compute */
    uint32_t runSteps;

    /* period for checkpoint creation */
    uint32_t checkpointPeriod;

    /* common directory for checkpoints */
    std::string checkpointDirectory;

    /* number of checkpoints written */
    uint32_t numCheckpoints;

    /* checkpoint step to restart from */
    int32_t restartStep;

    /* common directory for restarts */
    std::string restartDirectory;

    /* restart requested */
    bool restartRequested;

    /* filename for checkpoint master file with all checkpoint timesteps */
    const std::string CHECKPOINT_MASTER_FILE;

private:

    /**
     * Set how often the elapsed time is printed.
     *
     * @param percent percentage difference for printing
     */
    void calcProgress()
    {
        if (progress == 0 || progress > 100)
            progress = 100;

        showProgressAnyStep = (uint32_t) ((double) runSteps / 100. * (double) progress);
        if (showProgressAnyStep == 0)
            showProgressAnyStep = 1;
    }

    /**
     * Append \p checkpointStep to the master checkpoint file
     *
     * @param checkpointStep current checkpoint step
     */
    void writeCheckpointStep(const uint32_t checkpointStep)
    {
        std::ofstream file;
        const std::string checkpointMasterFile =
            checkpointDirectory + std::string("/") + CHECKPOINT_MASTER_FILE;

        file.open(checkpointMasterFile.c_str(), std::ofstream::app);

        if (!file)
            throw std::runtime_error("Failed to write checkpoint master file");

        file << checkpointStep << std::endl;
        file.close();
    }

    bool output;

    uint16_t progress;
    uint32_t showProgressAnyStep;

    TimeIntervall tSimulation;
    TimeIntervall tInit;

};
} // namespace PMacc

