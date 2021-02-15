/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Rene Widera, Alexander Debus,
 *                     Benjamin Worpitz, Alexander Grund
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

#include "pmacc/types.hpp"

#include "pmacc/mappings/simulation/GridController.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "TimeInterval.hpp"
#include "pmacc/dataManagement/DataConnector.hpp"
#include "pmacc/Environment.hpp"
#include "pmacc/pluginSystem/IPlugin.hpp"
#include "pmacc/pluginSystem/containsStep.hpp"
#include "pmacc/pluginSystem/toTimeSlice.hpp"

#include <boost/filesystem.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
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
        using SeqOfTimeSlices = std::vector<pluginSystem::TimeSlice>;

        /**
         * Constructor
         *
         */
        SimulationHelper()
            : runSteps(0)
            , checkpointDirectory("checkpoints")
            , numCheckpoints(0)
            , restartStep(-1)
            , restartDirectory("checkpoints")
            , restartRequested(false)
            , CHECKPOINT_MASTER_FILE("checkpoints.txt")
            , author("")
            , useMpiDirect(false)
        {
            tSimulation.toggleStart();
            tInit.toggleStart();
        }

        virtual ~SimulationHelper()
        {
            tSimulation.toggleEnd();
            if(output)
            {
                std::cout << "full simulation time: " << tSimulation.printInterval() << " = "
                          << (uint64_t)(tSimulation.getInterval() / 1000.) << " sec" << std::endl;
            }
        }

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

        /**
         * Notifies registered output classes.
         *
         * This function is called automatically.
         *
         *  @param currentStep simulation step
         */
        virtual void dumpOneStep(uint32_t currentStep)
        {
            /* trigger notification */
            Environment<DIM>::get().PluginConnector().notifyPlugins(currentStep);

            /* trigger checkpoint notification */
            if(!checkpointPeriod.empty() && pluginSystem::containsStep(seqCheckpointPeriod, currentStep))
            {
                /* first synchronize: if something failed, we can spare the time
                 * for the checkpoint writing */
                CUDA_CHECK(cuplaDeviceSynchronize());
                CUDA_CHECK(cuplaGetLastError());

                // avoid deadlock between not finished PMacc tasks and MPI_Barrier
                __getTransactionEvent().waitForFinished();

                GridController<DIM>& gc = Environment<DIM>::get().GridController();
                /* can be spared for better scalings, but allows to spare the
                 * time for checkpointing if some ranks died */
                MPI_CHECK(MPI_Barrier(gc.getCommunicator().getMPIComm()));

                /* create directory containing checkpoints  */
                if(numCheckpoints == 0)
                {
                    Environment<DIM>::get().Filesystem().createDirectoryWithPermissions(checkpointDirectory);
                }

                Environment<DIM>::get().PluginConnector().checkpointPlugins(currentStep, checkpointDirectory);

                /* important synchronize: only if no errors occured until this
                 * point guarantees that a checkpoint is usable */
                CUDA_CHECK(cuplaDeviceSynchronize());
                CUDA_CHECK(cuplaGetLastError());

                /* avoid deadlock between not finished PMacc tasks and MPI_Barrier */
                __getTransactionEvent().waitForFinished();

                /* \todo in an ideal world with MPI-3, this would be an
                 * MPI_Ibarrier call and this function would return a MPI_Request
                 * that could be checked */
                MPI_CHECK(MPI_Barrier(gc.getCommunicator().getMPIComm()));

                if(gc.getGlobalRank() == 0)
                {
                    writeCheckpointStep(currentStep);
                }
                numCheckpoints++;
            }
        }

        GridController<DIM>& getGridController()
        {
            return Environment<DIM>::get().GridController();
        }

        void dumpTimes(TimeIntervall& tSimCalculation, TimeIntervall&, double& roundAvg, uint32_t currentStep)
        {
            /*dump 100% after simulation*/
            if(output && progress && (currentStep % showProgressAnyStep) == 0)
            {
                tSimCalculation.toggleEnd();
                std::cout << std::setw(3)
                          << uint16_t(
                                 double(currentStep)
                                 / double(Environment<>::get().SimulationDescription().getRunSteps()) * 100.)
                          << " % = " << std::setw(8) << currentStep << " | time elapsed:" << std::setw(25)
                          << tSimCalculation.printInterval() << " | avg time per step: "
                          << TimeIntervall::printeTime(roundAvg / (double) showProgressAnyStep) << std::endl;
                std::cout.flush();

                roundAvg = 0.0; // clear round avg timer
            }
        }

        /**
         * Begin the simulation.
         */
        void startSimulation()
        {
            if(useMpiDirect)
                Environment<>::get().enableMpiDirect();

            init();

            // translate checkpointPeriod string into checkpoint intervals
            seqCheckpointPeriod = pluginSystem::toTimeSlice(checkpointPeriod);

            for(uint32_t nthSoftRestart = 0; nthSoftRestart <= softRestarts; ++nthSoftRestart)
            {
                resetAll(0);
                uint32_t currentStep = fillSimulation();
                Environment<>::get().SimulationDescription().setCurrentStep(currentStep);

                tInit.toggleEnd();
                if(output)
                {
                    std::cout << "initialization time: " << tInit.printInterval() << " = "
                              << (int) (tInit.getInterval() / 1000.) << " sec" << std::endl;
                }

                TimeIntervall tSimCalculation;
                TimeIntervall tRound;
                double roundAvg = 0.0;

                /* Since in the main loop movingWindow is called always before the dump, we also call it here for
                 * consistency. This becomes only important, if movingWindowCheck does more than merely checking for a
                 * slide. TO DO in a new feature: Turn this into a general hook for pre-checks (window slides are just
                 * one possible action).
                 */
                movingWindowCheck(currentStep);

                /* dump initial step if simulation starts without restart */
                if(!restartRequested)
                {
                    dumpOneStep(currentStep);
                }

                /* dump 0% output */
                dumpTimes(tSimCalculation, tRound, roundAvg, currentStep);


                /** \todo currently we assume this is the only point in the simulation
                 *        that is allowed to manipulate `currentStep`. Else, one needs to
                 *        add and act on changed values via
                 *        `SimulationDescription().getCurrentStep()` in this loop
                 */
                while(currentStep < Environment<>::get().SimulationDescription().getRunSteps())
                {
                    tRound.toggleStart();
                    runOneStep(currentStep);
                    tRound.toggleEnd();
                    roundAvg += tRound.getInterval();

                    /* NEXT TIMESTEP STARTS HERE */
                    currentStep++;
                    Environment<>::get().SimulationDescription().setCurrentStep(currentStep);
                    /* output times after a round */
                    dumpTimes(tSimCalculation, tRound, roundAvg, currentStep);

                    movingWindowCheck(currentStep);
                    /* dump at the beginning of the simulated step */
                    dumpOneStep(currentStep);
                }

                // simulatation end
                Environment<>::get().Manager().waitForAllTasks();

                tSimCalculation.toggleEnd();

                if(output)
                {
                    std::cout << "calculation  simulation time: " << tSimCalculation.printInterval() << " = "
                              << (int) (tSimCalculation.getInterval() / 1000.) << " sec" << std::endl;
                }

            } // softRestarts loop
        }

        virtual void pluginRegisterHelp(po::options_description& desc)
        {
            desc.add_options()("steps,s", po::value<uint32_t>(&runSteps), "Simulation steps")(
                "checkpoint.restart.loop",
                po::value<uint32_t>(&softRestarts)->default_value(0),
                "Number of times to restart the simulation after simulation has finished (for presentations). "
                "Note: does not yet work with all plugins, see issue #1305")(
                "percent,p",
                po::value<uint16_t>(&progress)->default_value(5),
                "Print time statistics after p percent to stdout")(
                "checkpoint.restart",
                po::value<bool>(&restartRequested)->zero_tokens(),
                "Restart simulation")(
                "checkpoint.restart.directory",
                po::value<std::string>(&restartDirectory)->default_value(restartDirectory),
                "Directory containing checkpoints for a restart")(
                "checkpoint.restart.step",
                po::value<int32_t>(&restartStep),
                "Checkpoint step to restart from")(
                "checkpoint.period",
                po::value<std::string>(&checkpointPeriod),
                "Period for checkpoint creation")(
                "checkpoint.directory",
                po::value<std::string>(&checkpointDirectory)->default_value(checkpointDirectory),
                "Directory for checkpoints")(
                "author",
                po::value<std::string>(&author)->default_value(std::string("")),
                "The author that runs the simulation and is responsible for created output files")(
                "mpiDirect",
                po::value<bool>(&useMpiDirect)->zero_tokens(),
                "use device direct for MPI communication e.g. GPU direct");
        }

        std::string pluginGetName() const
        {
            return "SimulationHelper";
        }

        void pluginLoad()
        {
            Environment<>::get().SimulationDescription().setRunSteps(runSteps);
            Environment<>::get().SimulationDescription().setAuthor(author);

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

        /** Presentations: loop the whole simulation `softRestarts` times from
         *                 initial step to runSteps */
        uint32_t softRestarts;

        /* period for checkpoint creation */
        std::string checkpointPeriod;

        /* checkpoint intervals */
        SeqOfTimeSlices seqCheckpointPeriod;

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

        /* author that runs the simulation */
        std::string author;

        //! enable MPI gpu direct
        bool useMpiDirect;

    private:
        /**
         * Set how often the elapsed time is printed.
         *
         * @param percent percentage difference for printing
         */
        void calcProgress()
        {
            if(progress == 0 || progress > 100)
                progress = 100;

            showProgressAnyStep = uint32_t(
                double(Environment<>::get().SimulationDescription().getRunSteps()) / 100. * double(progress));
            if(showProgressAnyStep == 0)
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
            const std::string checkpointMasterFile = checkpointDirectory + std::string("/") + CHECKPOINT_MASTER_FILE;

            file.open(checkpointMasterFile.c_str(), std::ofstream::app);

            if(!file)
                throw std::runtime_error("Failed to write checkpoint master file");

            file << checkpointStep << std::endl;
            file.close();
        }

    protected:
        /**
         * Reads the checkpoint master file if any and returns all found checkpoint steps
         *
         * @return vector of found checkpoints steps in order they appear in the file
         */
        std::vector<uint32_t> readCheckpointMasterFile()
        {
            std::vector<uint32_t> checkpoints;

            const std::string checkpointMasterFile
                = this->restartDirectory + std::string("/") + this->CHECKPOINT_MASTER_FILE;

            if(!boost::filesystem::exists(checkpointMasterFile))
                return checkpoints;

            std::ifstream file(checkpointMasterFile.c_str());

            /* read each line */
            std::string line;
            while(std::getline(file, line))
            {
                if(line.empty())
                    continue;
                try
                {
                    checkpoints.push_back(boost::lexical_cast<uint32_t>(line));
                }
                catch(boost::bad_lexical_cast const&)
                {
                    std::cerr << "Warning: checkpoint master file contains invalid data (" << line << ")" << std::endl;
                }
            }

            return checkpoints;
        }

    private:
        bool output = false;

        uint16_t progress;
        uint32_t showProgressAnyStep;

        TimeIntervall tSimulation;
        TimeIntervall tInit;
    };

} // namespace pmacc
