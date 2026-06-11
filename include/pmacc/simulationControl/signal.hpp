/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

namespace pmacc
{
    namespace signal
    {
        namespace detail
        {
            void setCreateCheckpoint(int signal);

            void setStopSimulation(int signal);

            void setCreateCheckpointAndStopSimulation(int signal);
        } // namespace detail

        /** Activate signal handling.
         *
         * @attention  Signals will not be registered on Windows operating system.
         * This function is in this cas empty.
         */
        void activateSignalHandling();

        /** Check if a signal was received
         *
         * @return true if at least one signal is received else false.
         *         If true is returned once this function is returning false until release() is called.
         */
        bool received();

        /** Release signals
         *
         * This function should only be called if received() returned true.
         *
         * @param checkPointHandled if true the checkpoint signal state is reset.
         * @param stopSimulationHandled if true the stop simulation signal state is reset.
         */
        void release(bool checkPointHandled, bool stopSimulationHandled);

        /** Status if checkpoint creation is requested.
         *
         * Status is resetting with each query.
         *
         * @return true if a checkpoint should be created else false.
         */
        bool createCheckpoint();

        /** Status if should be stopped.
         *
         * Status is resetting with each query.
         *
         * @return true should be stopped else false
         */
        bool stopSimulation();

    } // namespace signal
} // namespace pmacc
