/**
 * Copyright 2013-2014 Axel Huebl, Felix Schmitt, Rene Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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

#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>

#include "types.h"

#include "mappings/simulation/GridController.hpp"
#include "dimensions/DataSpace.hpp"
#include "TimeInterval.hpp"

#include "dataManagement/DataConnector.hpp"


#include "eventSystem/EventSystem.hpp"
#include "pluginSystem/IPlugin.hpp"


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
    restartRequested(false)
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
        //CUDA_CHECK(cudaGetLastError());
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
        
        /* trigger checkpoint notification first to allow plugins to skip standard notify */
        if (checkpointPeriod && (currentStep % checkpointPeriod == 0))
        {
            /* create directory containing checkpoints  */
            if (numCheckpoints == 0)
            {
                boost::filesystem::create_directories(checkpointDirectory);
            }
            
            Environment<DIM>::get().PluginConnector().checkpointPlugins(currentStep,
                                                                        checkpointDirectory);
            numCheckpoints++;
        }
        
        Environment<DIM>::get().PluginConnector().notifyPlugins(currentStep);
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

        TimeIntervall tSimCalculation;
        TimeIntervall tRound;
        double roundAvg = 0.0;

        /* dump initial step if simulation starts without restart */            
        if (currentStep == 0)
            dumpOneStep(currentStep);
        else
            currentStep--; //we dump before calculation, thus we must go on step back if we do a restart

        movingWindowCheck(currentStep); //if we restart at any step check if we must slide

        /* dump 0% output */
        dumpTimes(tSimCalculation,tRound,roundAvg,currentStep);
        while (currentStep < runSteps)
        {
            tRound.toggleStart();
            runOneStep(currentStep);
            tRound.toggleEnd();
            roundAvg += tRound.getInterval();

            currentStep++;
            /*output after a round*/
            dumpTimes(tSimCalculation,tRound,roundAvg,currentStep);
            
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

    bool output;

    uint16_t progress;
    uint32_t showProgressAnyStep;

    TimeIntervall tSimulation;
    TimeIntervall tInit;

};
} // namespace PMacc

