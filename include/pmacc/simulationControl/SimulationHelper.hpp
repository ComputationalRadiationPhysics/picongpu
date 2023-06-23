/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Rene Widera, Alexander Debus,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov, Pawel Ordyna
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "TimeInterval.hpp"
#include "pmacc/Environment.hpp"
#include "pmacc/mappings/simulation/GridController.hpp"
#include "pmacc/pluginSystem/IPlugin.hpp"
#include "pmacc/types.hpp"

#include <string>
#include <vector>


namespace pmacc
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
        using SeqOfTimeSlices = std::vector<pluginSystem::Slice>;

        /**
         * Constructor
         *
         */
        SimulationHelper();

        ~SimulationHelper() override;

        /**
         * Must describe one iteration (step).
         *
         * This function is called automatically.
         */
        virtual void runOneStep(uint32_t currentStep) = 0;

        /**
         * Initialize simulation
         *
         * Does hardware selections/reservations, memory allocations and
         * initializes data structures as empty.
         */
        virtual void init() = 0;

        /**
         * Fills simulation with initial data after init()
         *
         * @return returns the first step of the simulation
         *         (can be >0 for, e.g., restarts from checkpoints)
         */
        virtual uint32_t fillSimulation() = 0;

        /**
         * Reset the simulation to a state such as it was after
         * init() but for a specific time step.
         * Can be used to call fillSimulation() again.
         */
        virtual void resetAll(uint32_t currentStep) = 0;

        /**
         * Check if moving window work must do
         *
         * If no moving window is needed the implementation of this function can be empty
         *
         * @param currentStep simulation step
         */
        virtual void movingWindowCheck(uint32_t currentStep) = 0;

        /** Call all plugins
         *
         * This function is called inside the simulation loop.
         *
         * @param currentStep simulation step
         */
        void notifyPlugins(uint32_t currentStep);

        /** Write a checkpoint if needed for the given step
         *
         * This function is called inside the simulation loop.
         *
         *  @param currentStep simulation step
         */
        virtual void dumpOneStep(uint32_t currentStep);

        GridController<DIM>& getGridController()
        {
            return Environment<DIM>::get().GridController();
        }

        void dumpTimes(TimeIntervall& tSimCalculation, TimeIntervall&, double& roundAvg, uint32_t currentStep);

        /**
         * Begin the simulation.
         */
        void startSimulation();

        void pluginRegisterHelp(po::options_description& desc) override;

        std::string pluginGetName() const override
        {
            return "SimulationHelper";
        }

        void pluginLoad() override;

        void pluginUnload() override
        {
        }

        void restart(uint32_t, const std::string) override
        {
        }

        void checkpoint(uint32_t, const std::string) override
        {
        }

    protected:
        /* number of simulation steps to compute */
        uint32_t runSteps{0};

        /** Presentations: loop the whole simulation `softRestarts` times from
         *                 initial step to runSteps */
        uint32_t softRestarts;

        /* period for checkpoint creation [interval(s) based on steps]*/
        std::string checkpointPeriod;

        /* checkpoint intervals */
        SeqOfTimeSlices seqCheckpointPeriod;

        /* period for checkpoint creation [period in minutes]
         * Zero is disabling time depended checkpointing.
         */
        std::uint64_t checkpointPeriodMinutes = 0u;
        std::thread checkpointTimeThread;

        // conditional variable to notify all concurrent threads and signal exit of the simulation
        std::condition_variable exitConcurrentThreads;

        /* common directory for checkpoints */
        std::string checkpointDirectory;

        /* number of checkpoints written */
        uint32_t numCheckpoints{0};

        /* checkpoint step to restart from */
        int32_t restartStep{-1};

        /* common directory for restarts */
        std::string restartDirectory;

        /* restart requested */
        bool restartRequested{false};

        /* filename for checkpoint master file with all checkpoint timesteps */
        const std::string CHECKPOINT_MASTER_FILE;

        /* author that runs the simulation */
        std::string author;

        //! enable MPI gpu direct
        bool useMpiDirect{false};

        bool tryRestart = false;

    private:
        /** Largest time step within the simulation (all MPI ranks) */
        uint32_t signalMaxTimestep = 0u;
        /** Time step at which we create actions out of an signal.*/
        uint32_t handleSignalAtStep = 0u;
        /** MPI request to find largest time step in the simulation */
        MPI_Request signalMPI = MPI_REQUEST_NULL;
        bool signalCreateCheckpoint = false;
        bool signalStopSimulation = false;

        void checkSignals(uint32_t const currentStep);

        /**
         * Set how often the elapsed time is printed.
         *
         * @param percent percentage difference for printing
         */
        void calcProgress();


        /**
         * Append \p checkpointStep to the master checkpoint file
         *
         * @param checkpointStep current checkpoint step
         */
        void writeCheckpointStep(const uint32_t checkpointStep);

    protected:
        /**
         * Reads the checkpoint master file if any and returns all found checkpoint steps
         *
         * @return vector of found checkpoints steps in order they appear in the file
         */
        std::vector<uint32_t> readCheckpointMasterFile();

    private:
        bool output = false;

        uint16_t progress;
        uint32_t showProgressAnyStep;

        /* progress intervals */
        bool progressStepPeriodEnabled = false;
        SeqOfTimeSlices seqProgressPeriod;
        std::string progressPeriod;
        uint32_t lastProgressStep = 0u;

        TimeIntervall tSimulation;
        TimeIntervall tInit;
    };

} // namespace pmacc
