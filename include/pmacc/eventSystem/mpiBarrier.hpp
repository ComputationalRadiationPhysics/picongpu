/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include <mpi.h>

namespace pmacc::eventSystem
{
    /** MPI Barrier
     *
     * The function is executing an MPI barrier while guaranteeing that the event system is not blocked.
     * You should call this function before you use MPI collective operations in your code to avoid deadlocks.
     * After the function returned you know that all participating MPI ranks reached this code line.
     *
     * @attention This function should be called from all MPI ranks within the communicator
     * This method is **NOT** waiting until all events in the event queue are processed.
     *
     * @param communicator communicator used for the barrier operation
     */
    void mpiBlocking(MPI_Comm communicator);
} // namespace pmacc::eventSystem
