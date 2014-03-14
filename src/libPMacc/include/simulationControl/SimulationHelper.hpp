/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Rene Widera
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
 

#ifndef SIMULATIONHELPER_HPP
#define	SIMULATIONHELPER_HPP

#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>

#include "types.h"

#include "mappings/simulation/GridController.hpp"
#include "dimensions/DataSpace.hpp"
#include "TimeInterval.hpp"

#include "dataManagement/DataConnector.hpp"


#include "eventSystem/EventSystem.hpp"
#include "moduleSystem/Module.hpp"


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
class SimulationHelper : public Module
{
public:

    /**
     * Constructor
     * 
     */
    SimulationHelper()
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
        Environment<DIM>::get().DataConnector().dumpData(currentStep);
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

        /*dump initial step if simulation start without restart*/
        if (currentStep == 0)
            dumpOneStep(currentStep);
        else
            currentStep--; //we dump before calculation, thus we must go on step back if we do a restart

        movingWindowCheck(currentStep); //if we restart at any step check if we must slide

        /*dum 0% output*/
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

    virtual void moduleRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ("steps,s", po::value<uint32_t > (&runSteps), "simulation steps")
            ("percent,p", po::value<uint16_t > (&progress)->default_value(5),
             "print time statistics after p percent to stdout");
    }

    std::string moduleGetName() const
    {
        return "SimulationHelper";
    }

    virtual void moduleLoad()
    {
        calcProgress();

        output = (getGridController().getGlobalRank() == 0);
    }

    virtual void moduleUnload()
    {
    }

protected:
    //! how much time steps shall be calculated
    uint32_t runSteps;

private:

    /**
     * Set how often the elapsed time is printed.
     * 
     * @param percent percentage difference for printing
     */
    void calcProgress()
    {
        if (progress == 0)
            progress = 100;
        else
            if (progress > 100)
            progress = 100;
        showProgressAnyStep = (uint32_t) ((double) runSteps / 100. * (double) progress);
        if (showProgressAnyStep == 0)
            showProgressAnyStep = 1;
    }

    //! how often calculated data will be dumped (picture or other format)
    uint32_t dumpAnyStep;

    bool output;

    uint16_t progress;
    uint32_t showProgressAnyStep;

    TimeIntervall tSimulation;
    TimeIntervall tInit;

};
} // namespace PMacc


#endif	/* SIMULATIONHELPER_HPP */

