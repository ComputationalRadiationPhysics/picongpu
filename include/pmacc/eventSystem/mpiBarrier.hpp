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
