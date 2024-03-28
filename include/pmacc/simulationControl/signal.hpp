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
        void activate();

        /** Check if a signal is received
         *
         * @return true if at least one signal is received else false
         */
        bool received();

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
