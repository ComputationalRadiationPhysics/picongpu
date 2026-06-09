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


#include "pmacc/eventSystem/mpiBarrier.hpp"

#include "pmacc/communication/manager_common.hpp"
#include "pmacc/eventSystem/Manager.hpp"

namespace pmacc::eventSystem
{
    void mpiBlocking(MPI_Comm communicator)
    {
        MPI_Request ioBarrierMPI = MPI_REQUEST_NULL;
        MPI_CHECK(MPI_Ibarrier(communicator, &ioBarrierMPI));
        // block until all MPI ranks reach the barrier but keep the event system active
        Manager::getInstance().waitFor(
            [&]() -> bool
            {
                MPI_Status mpiBarrierStatus;
                int flag = 0;
                MPI_CHECK(MPI_Test(&ioBarrierMPI, &flag, &mpiBarrierStatus));
                return flag != 0;
            });
    }
} // namespace pmacc::eventSystem
