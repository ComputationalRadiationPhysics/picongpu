/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/* Copyright 2026 Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/communication/manager_common.hpp"
#include "pmacc/simulationControl/signal.hpp"

#include <mpi.h>

namespace pmacc
{
    template<unsigned T_dim, typename T_CheckPointing>
    class Signal final : public ITask
    {
    public:
        Signal(uint32_t currentStep, T_CheckPointing& checkpointing, bool writeOutput)
            : m_checkpointing(checkpointing)
            , m_writeOutput(writeOutput)
        {
            if(signal::received())
            {
                /* Set to the next possible step we could execute checkpointing or stop the simulation
                 * All ranks will agree for a common time step which is currently not processed yet.
                 */
                m_processSignalAtStep = currentStep + 1;
                m_state = Init;

                if(m_writeOutput)
                    std::cout << "SIGNAL: received." << std::endl;
            }
        }

        void init() override
        {
            if(m_state == Init)
            {
                // find the largest time step of all MPI ranks
                MPI_CHECK(MPI_Iallreduce(
                    &m_processSignalAtStep,
                    &m_globalCommonTimestep,
                    1,
                    MPI_UINT32_T,
                    MPI_MAX,
                    Environment<T_dim>::get().GridController().getCommunicator().getMPISignalComm(),
                    &m_reduceTimeStepRequest));


                m_sendSignals[doCheckpointing] = signal::createCheckpoint();
                m_sendSignals[stopSimulation] = signal::stopSimulation();

                MPI_CHECK(MPI_Iallreduce(
                    m_sendSignals.data(),
                    m_globalSignalCounts.data(),
                    m_globalSignalCounts.size(),
                    MPI_UINT32_T,
                    MPI_SUM,
                    Environment<T_dim>::get().GridController().getCommunicator().getMPISignalComm(),
                    &m_signalRequest));

                m_state = WaitForMpiReduce;
            }
        }

        bool executeIntern() override
        {
            if(m_state == Finished)
                return true;

            if(m_state == WaitForMpiReduce)
            {
                if(m_reduceTimeStepRequest != MPI_REQUEST_NULL)
                {
                    // wait for the global common time step
                    MPI_Status mpiTimeStepStatus;

                    int flag = 0;
                    MPI_CHECK(MPI_Test(&m_reduceTimeStepRequest, &flag, &mpiTimeStepStatus));
                    if(flag != 0)
                        m_reduceTimeStepRequest = MPI_REQUEST_NULL;
                }
                else if(m_signalRequest != MPI_REQUEST_NULL)
                {
                    // wait for signal categories
                    MPI_Status mpiSignalStatus;

                    int flag = 0;
                    MPI_CHECK(MPI_Test(&m_signalRequest, &flag, &mpiSignalStatus));
                    if(flag != 0)
                        m_signalRequest = MPI_REQUEST_NULL;
                }
                // wait until we know the max timestep of all MPI ranks and which signals we should handle
                if(m_reduceTimeStepRequest == MPI_REQUEST_NULL && m_signalRequest == MPI_REQUEST_NULL)
                    m_state = HandleSignals;
            }

            if(m_state == HandleSignals)
            {
                uint32_t numMpiRanks = Environment<T_dim>::get().GridController().getCommunicator().getSize();
                /* Only if all MPI ranks see the same signal category we can apply the corresponding action.
                 * Later we release only those categories every MPI ranks processed, all not processed categories will
                 * be handled with the next TaskSignal.
                 */
                bool shouldCreateCheckpoint = m_globalSignalCounts[doCheckpointing] == numMpiRanks;
                bool shouldStop = m_globalSignalCounts[stopSimulation] == numMpiRanks;

                // Translate signals into actions
                if(shouldCreateCheckpoint)
                {
                    if(m_writeOutput)
                        std::cout << "SIGNAL: Activate checkpointing for step " << m_globalCommonTimestep << std::endl;

                    // add a new checkpoint
                    m_checkpointing.addCheckpoint(m_globalCommonTimestep);
                }

                if(shouldStop)
                {
                    if(m_writeOutput)
                        std::cout << "SIGNAL: Shutdown simulation at step " << m_globalCommonTimestep << std::endl;

                    Environment<>::get().SimulationDescription().setRunSteps(m_globalCommonTimestep);
                }

                /** @attention If we miss releasing the signal system we will never create a TaskSignal again and can
                 * not handle signals anymore. */
                signal::release(shouldCreateCheckpoint, shouldStop);
                m_state = Finished;
                return true;
            }

            return false;
        }

        ~Signal() override
        {
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
        {
            return std::string("Signal at stage") + std::to_string(m_state);
        }

    private:
        /** Instance where a checkpoint can be registered to. */
        T_CheckPointing& m_checkpointing;
        /** The time step in which this MPI rank would like to apply actions based on the signal.
         *
         * atomic is not required because the event system is not threaded
         */
        uint32_t m_processSignalAtStep = 0u;
        /** Largest common timestep within all MPI ranks */
        uint32_t m_globalCommonTimestep = 0u;
        /** Number of MPI ranks received the signal to stop the simulation */
        uint32_t m_globalNumStopSignals = 0u;
        /** Number of MPI ranks received the signal to checkpoint the simulation */
        uint32_t m_globalNumCheckpointSignals = 0u;

        /** MPI Request to check the status for the common time step MPI call */
        MPI_Request m_reduceTimeStepRequest = MPI_REQUEST_NULL;
        /** MPI Request to check the status for the common category MPI call */
        MPI_Request m_signalRequest = MPI_REQUEST_NULL;

        /** Signal categories to send
         *
         * To access slots you should use SignalType
         */
        std::array<uint32_t, 2u> m_sendSignals = {0u, 0u};
        /** Aggregated results of all MPI ranks
         *
         * Each component contains the number of ranks seeing the corresponding signal category.
         */
        std::array<uint32_t, 2u> m_globalSignalCounts = {0u, 0u};

        enum StateType
        {
            Finished,
            Init,
            WaitForMpiReduce,
            HandleSignals
        };

        enum SignalType
        {
            stopSimulation,
            doCheckpointing
        };

        StateType m_state = Finished;
        bool m_writeOutput = false;
    };

} // namespace pmacc
