/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#include <atomic>
#include <csignal>

namespace pmacc
{
    namespace signal
    {
        namespace detail
        {
            namespace
            {
                std::atomic<int> gStatusCreateCheckpoint = 0;
                std::atomic<int> gStatusStopSimulation = 0;
                std::atomic<int> gSignalLocked = 0;
            } // namespace

            void setCreateCheckpoint(int)
            {
                gStatusCreateCheckpoint += 1;
            }

            void setStopSimulation(int)
            {
                gStatusStopSimulation += 1;
            }

            void setCreateCheckpointAndStopSimulation(int)
            {
                gStatusCreateCheckpoint += 1;
                gStatusStopSimulation += 1;
            }
        } // namespace detail

        void activateSignalHandling()
        {
#ifndef _WIN32
            std::signal(SIGHUP, detail::setStopSimulation);
            std::signal(SIGINT, detail::setStopSimulation);
            std::signal(SIGQUIT, detail::setStopSimulation);
            std::signal(SIGABRT, detail::setStopSimulation);
            std::signal(SIGUSR1, detail::setCreateCheckpoint);
            std::signal(SIGUSR2, detail::setStopSimulation);
            std::signal(SIGALRM, detail::setCreateCheckpointAndStopSimulation);
            std::signal(SIGTERM, detail::setStopSimulation);
#endif
        }

        bool received()
        {
            /* Do not handle any new signals until the last signal handle step is finished.
             * Whoever called received() and got true must call release() after handling the signal.
             */
            if(detail::gSignalLocked)
                return false;

            bool receivedASignal
                = detail::gStatusCreateCheckpoint.load() != 0 || detail::gStatusStopSimulation.load() != 0;
            if(receivedASignal)
                detail::gSignalLocked = 1;
            return receivedASignal;
        }

        void release(bool checkPointHandled, bool stopSimulationHandled)
        {
            // only reset a signal if the signal counters are locked
            if(detail::gSignalLocked)
            {
                if(checkPointHandled)
                    detail::gStatusCreateCheckpoint -= 1;
                if(stopSimulationHandled)
                    detail::gStatusStopSimulation -= 1;
            }
            detail::gSignalLocked = 0;
        }

        bool createCheckpoint()
        {
            auto checkPointRequested = detail::gStatusCreateCheckpoint.load() != 0;
            return checkPointRequested;
        }

        bool stopSimulation()
        {
            // do not stop the simulation as long there are more than one outstanding checkpoint signal which are not
            // handled yet
            auto stopRequested
                = detail::gStatusStopSimulation.load() != 0 && detail::gStatusCreateCheckpoint.load() <= 1;
            return stopRequested;
        }
    } // namespace signal
} // namespace pmacc
