/* Copyright 2021-2023 Rene Widera
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


            void setCreateCheckpoint(int signal)
            {
                gStatusCreateCheckpoint = 1;
            }

            void setStopSimulation(int signal)
            {
                gStatusStopSimulation = 1;
            }

            void setCreateCheckpointAndStopSimulation(int signal)
            {
                gStatusCreateCheckpoint = 1;
                gStatusStopSimulation = 1;
            }
        } // namespace detail

        void activate()
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
            return detail::gStatusCreateCheckpoint != 0 || detail::gStatusStopSimulation != 0;
        }


        bool createCheckpoint()
        {
            auto result = detail::gStatusCreateCheckpoint != 0;
            if(result)
                detail::gStatusCreateCheckpoint = 0;
            return result;
        }

        bool stopSimulation()
        {
            auto result = detail::gStatusStopSimulation != 0;
            if(result)
                detail::gStatusCreateCheckpoint = 0;
            return result;
        }
    } // namespace signal
} // namespace pmacc
