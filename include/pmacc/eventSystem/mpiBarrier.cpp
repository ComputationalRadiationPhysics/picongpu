/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
