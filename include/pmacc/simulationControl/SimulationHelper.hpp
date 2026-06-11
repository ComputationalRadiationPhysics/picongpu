/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Rene Widera, Alexander Debus, Benjamin Worpitz, Alexander Grund, Sergei Bastrakov, Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "TimeInterval.hpp"
#include "pmacc/Environment.hpp"
#include "pmacc/mappings/simulation/GridController.hpp"
#include "pmacc/pluginSystem/IPlugin.hpp"
#include "pmacc/pluginSystem/Slice.hpp"
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
    template<unsigned DIM, typename CheckpointingClass>
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
        virtual void notifyPlugins(uint32_t currentStep);

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

        void dumpTimes(TimeInterval& tSimCalculation, TimeInterval&, double& roundAvg, uint32_t currentStep);

        /**
         * Begin the simulation.
         */
        virtual void startSimulation();

        void pluginRegisterHelp(po::options_description& desc) override;

        std::string pluginGetName() const override
        {
            return "SimulationHelper";
        }

        void pluginLoad() override;

        void pluginUnload() override
        {
        }

        void restart(uint32_t, std::string const) override
        {
        }

        void checkpoint(uint32_t, std::string const) override
        {
        }

    protected:
        /* number of simulation steps to compute */
        uint32_t runSteps{0};

        CheckpointingClass checkpointing;
        /* author that runs the simulation */
        std::string author;

        //! enable MPI gpu direct
        bool useMpiDirect{false};


    private:
        /** Checks if we received a signal.
         *
         * This method can be called multiple time within a timestep to lower the latency of the response to it.
         * The call is cheap if no signal was received.
         * If signals of the same action will be sent multiple times they will be handled after ongoing registered
         * action resulting from previous signals are executed.
         */
        void checkSignals(uint32_t const currentStep);

        /**
         * Set how often the elapsed time is printed.
         *
         * @param percent percentage difference for printing
         */
        void calcProgress();

        bool output = false;

        uint16_t progress;
        uint32_t showProgressAnyStep;

        /* progress intervals */
        bool progressStepPeriodEnabled = false;
        SeqOfTimeSlices seqProgressPeriod;
        std::string progressPeriod;
        uint32_t lastProgressStep = 0u;

        TimeInterval tSimulation;
        TimeInterval tInit;
    };

} // namespace pmacc
