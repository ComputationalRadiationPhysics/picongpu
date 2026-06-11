/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Rene Widera, Alexander Debus, Benjamin Worpitz, Alexander Grund, Sergei Bastrakov, Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pmacc/simulationControl/SimulationHelper.hpp"

#include "TimeInterval.hpp"
#include "pmacc/Environment.hpp"
#include "pmacc/dataManagement/DataConnector.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/Manager.hpp"
#include "pmacc/particles/IdProvider.hpp"
#include "pmacc/pluginSystem/IPlugin.hpp"
#include "pmacc/pluginSystem/containsStep.hpp"
#include "pmacc/pluginSystem/toSlice.hpp"
#include "pmacc/simulationControl/Checkpointing.hpp"
#include "pmacc/simulationControl/signal.hpp"
#include "pmacc/types.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

namespace pmacc
{
    template<unsigned DIM, typename CheckpointingClass>
    SimulationHelper<DIM, CheckpointingClass>::SimulationHelper() : author("")
    {
        tSimulation.toggleStart();
        tInit.toggleStart();
    }

    template<unsigned DIM, typename CheckpointingClass>
    SimulationHelper<DIM, CheckpointingClass>::~SimulationHelper()
    {
        checkpointing.finishTimeBasedCheckpointing();
        tSimulation.toggleEnd();
        if(output)
        {
            std::cout << "full simulation time: " << tSimulation.printInterval() << " = " << std::fixed
                      << std::setprecision(3) << (tSimulation.getInterval() / 1000.)
                      << std::resetiosflags(std::ios::showbase) << " sec" << std::endl;
        }
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::notifyPlugins(uint32_t currentStep)
    {
        checkSignals(currentStep);
        Environment<DIM>::get().PluginConnector().notifyPlugins(currentStep);
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::dumpOneStep(uint32_t currentStep)
    {
        checkSignals(currentStep);
        checkpointing.template dump<DIM>(currentStep);
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::dumpTimes(
        TimeInterval& tSimCalculation,
        TimeInterval&,
        double& roundAvg,
        uint32_t currentStep)
    {
        /*dump 100% after simulation*/
        bool const writeAtPercent = progress && (currentStep % showProgressAnyStep) == 0;
        bool const writeAtStepPeriod
            = (progressStepPeriodEnabled && pluginSystem::containsStep(seqProgressPeriod, currentStep));
        if(output && (writeAtPercent || writeAtStepPeriod))
        {
            tSimCalculation.toggleEnd();
            uint32_t progressInterval{currentStep - lastProgressStep};
            // avoid division by 0 at 0% output
            if(progressInterval == 0u)
            {
                progressInterval = 1u;
            }
            std::cout << std::setw(3)
                      << uint16_t(
                             static_cast<double>(currentStep)
                             / static_cast<double>(Environment<>::get().SimulationDescription().getRunSteps()) * 100.)
                      << " % = " << std::setw(8) << currentStep << " | time elapsed:" << std::setw(25)
                      << tSimCalculation.printInterval() << " | avg time per step: "
                      << TimeInterval::printTime(roundAvg / static_cast<double>(progressInterval)) << std::endl;
            std::cout.flush();

            lastProgressStep = currentStep;
            roundAvg = 0.0; // clear round avg timer
        }
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::startSimulation()
    {
        if(useMpiDirect)
            Environment<>::get().enableMpiDirect();

        // Install a signal handler
        signal::activateSignalHandling();

        checkpointing.startTimeBasedCheckpointing();

        uint64_t maxRanks = Environment<DIM>::get().GridController().getGpuNodes().productOfComponents();
        uint64_t rank = Environment<DIM>::get().GridController().getScalarPosition();

        DataConnector& dc = Environment<>::get().DataConnector();
        auto idProvider = std::make_shared<IdProvider>("globalId", rank, maxRanks);
        dc.share(idProvider);

        init();

        while(checkpointing.hasSoftRestartAttemptsLeft())
        {
            /* Global offset is updated during the simulation. In case we perform a soft restart we need to reset
             * the offset here to be valid for the next simulation run.
             */
            Environment<DIM>::get().SubGrid().setGlobalDomainOffset(DataSpace<DIM>::create(0));
            resetAll(0);
            uint32_t currentStep = fillSimulation();
            Environment<>::get().SimulationDescription().setCurrentStep(currentStep);

            /* Ensure all ranks finished the initialization.
             * This synchronization costs a little bit time but possible errors during the initialization will be
             * easier to hunt because the rank that outputs timings will only show the timing for the initialization if
             * all ranks reached this point.
             */
            eventSystem::mpiBlocking(Environment<DIM>::get().GridController().getCommunicator().getMPIComm());

            tInit.toggleEnd();
            if(output)
            {
                std::cout << "initialization time: " << tInit.printInterval() << " = " << std::fixed
                          << std::setprecision(3) << (tInit.getInterval() / 1000.)
                          << std::resetiosflags(std::ios::showbase) << " sec" << std::endl;
            }

            TimeInterval tSimCalculation;
            TimeInterval tRound;
            double roundAvg = 0.0;

            /* Since in the main loop movingWindow is called always before the dump, we also call it here for
             * consistency. This becomes only important, if movingWindowCheck does more than merely checking for a
             * slide. TO DO in a new feature: Turn this into a general hook for pre-checks (window slides are just
             * one possible action).
             */
            movingWindowCheck(currentStep);

            /* call plugins and dump initial step if simulation starts without restart */
            if(checkpointing.getRestartState() != simulationControl::RestartState::SUCCESS)
            {
                notifyPlugins(currentStep);
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

                /* Next timestep starts here.
                 * Thus, for each timestep the plugins and checkpoint are called first.
                 * And the computational stages later on (on the next iteration of this loop).
                 */
                currentStep++;
                Environment<>::get().SimulationDescription().setCurrentStep(currentStep);
                /* output times after a round */
                dumpTimes(tSimCalculation, tRound, roundAvg, currentStep);

                movingWindowCheck(currentStep);
                /* call all plugins */
                notifyPlugins(currentStep);
                /* dump at the beginning of the simulated step */
                dumpOneStep(currentStep);
            }

            // The simulation is finished, wait until all MPI ranks finished the time step loop
            MPI_Request globalMPISync = MPI_REQUEST_NULL;
            MPI_CHECK(
                MPI_Ibarrier(Environment<DIM>::get().GridController().getCommunicator().getMPIComm(), &globalMPISync));
            Manager::getInstance().waitFor(
                [&]() -> bool
                {
                    // check for signal in case other MPI ranks still process the time step loop
                    checkSignals(currentStep);
                    MPI_Status mpiBarrierStatus;
                    int flag = 0;
                    MPI_CHECK(MPI_Test(&globalMPISync, &flag, &mpiBarrierStatus));
                    return flag != 0;
                });

            // ensure that the event system processed all tasks
            eventSystem::getTransactionEvent().waitForFinished();

            tSimCalculation.toggleEnd();

            if(output)
            {
                std::cout << "calculation  simulation time: " << tSimCalculation.printInterval() << " = " << std::fixed
                          << std::setprecision(3) << (tSimCalculation.getInterval() / 1000.)
                          << std::resetiosflags(std::ios::showbase) << " sec" << std::endl;
            }

        } // softRestarts loop
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::pluginRegisterHelp(po::options_description& desc)
    {
        // clang-format off
            desc.add_options()
                ("steps,s", po::value<uint32_t>(&runSteps), "Simulation steps")
                ("percent,p", po::value<uint16_t>(&progress)->default_value(5),
                 "Print time statistics after p percent to stdout")
                ("progressPeriod",po::value<std::string>(&progressPeriod),
                "write progress [for each n-th step], plugin period syntax can be used here.")
                ("author", po::value<std::string>(&author)->default_value(std::string("")),
                 "The author that runs the simulation and is responsible for created output files")
                ("mpiDirect", po::value<bool>(&useMpiDirect)->zero_tokens(),
                 "use device direct for MPI communication e.g. GPU direct");
        // clang-format on
        checkpointing.registerHelp(desc);
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::pluginLoad()
    {
        Environment<>::get().SimulationDescription().setRunSteps(runSteps);
        Environment<>::get().SimulationDescription().setAuthor(author);

        calcProgress();
        progressStepPeriodEnabled = !progressPeriod.empty();
        if(progressStepPeriodEnabled)
            seqProgressPeriod = pluginSystem::toTimeSlice(progressPeriod);

        output = (getGridController().getGlobalRank() == 0);
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::checkSignals(uint32_t const currentStep)
    {
        Environment<>::get().Factory().template createTaskSignal<DIM>(currentStep, checkpointing, output);
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::calcProgress()
    {
        if(progress == 0 || progress > 100)
            progress = 100;

        showProgressAnyStep
            = uint32_t(double(Environment<>::get().SimulationDescription().getRunSteps()) / 100. * double(progress));
        if(showProgressAnyStep == 0)
            showProgressAnyStep = 1;
    }


    // Explicit template instantiation to provide symbols for usage together with PMacc
    template class SimulationHelper<
        DIM1,
        simulationControl::Checkpointing<simulationControl::CheckpointingAvailability::ENABLED>>;
    template class SimulationHelper<
        DIM2,
        simulationControl::Checkpointing<simulationControl::CheckpointingAvailability::ENABLED>>;
    template class SimulationHelper<
        DIM3,
        simulationControl::Checkpointing<simulationControl::CheckpointingAvailability::ENABLED>>;
    template class SimulationHelper<
        DIM1,
        simulationControl::Checkpointing<simulationControl::CheckpointingAvailability::DISABLED>>;
    template class SimulationHelper<
        DIM2,
        simulationControl::Checkpointing<simulationControl::CheckpointingAvailability::DISABLED>>;
    template class SimulationHelper<
        DIM3,
        simulationControl::Checkpointing<simulationControl::CheckpointingAvailability::DISABLED>>;
} // namespace pmacc
