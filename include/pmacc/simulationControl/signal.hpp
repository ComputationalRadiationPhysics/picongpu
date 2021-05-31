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

#include <csignal>

namespace pmacc
{
    namespace signal
    {
        namespace detail
        {
            namespace
            {
                volatile std::sig_atomic_t gStatusCreateCheckpoint = 0;
                volatile std::sig_atomic_t gStatusStopSimulation = 0;
            } // namespace


            inline void setCreateCheckpoint(int signal)
            {
                gStatusCreateCheckpoint = 1;
            }

            inline void setStopSimulation(int signal)
            {
                gStatusStopSimulation = 1;
            }

            inline void setCreateCheckpointAndStopSimulation(int signal)
            {
                gStatusCreateCheckpoint = 1;
                gStatusStopSimulation = 1;
            }
        } // namespace detail

        /** Activate signal handling.
         *
         * @attention  Signals will not be registered on Windows operating system.
         * This function is in this cas empty.
         */
        inline void activate()
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

        /** Check if a signal is received
         *
         * @return true if at least one signal is received else false
         */
        inline bool received()
        {
            return detail::gStatusCreateCheckpoint != 0 || detail::gStatusStopSimulation != 0;
        }

        /** Status if checkpoint creation is requested.
         *
         * Status is resetting with each query.
         *
         * @return true if a checkpoint should be created else false.
         */
        inline bool createCheckpoint()
        {
            auto result = detail::gStatusCreateCheckpoint != 0;
            if(result)
                detail::gStatusCreateCheckpoint = 0;
            return result;
        }

        /** Status if should be stopped.
         *
         * Status is resetting with each query.
         *
         * @return true should be stopped else false
         */
        inline bool stopSimulation()
        {
            auto result = detail::gStatusStopSimulation != 0;
            if(result)
                detail::gStatusCreateCheckpoint = 0;
            return result;
        }
    } // namespace signal
} // namespace pmacc
