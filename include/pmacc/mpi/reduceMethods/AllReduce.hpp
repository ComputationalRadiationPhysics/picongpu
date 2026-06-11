/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/communication/manager_common.hpp"
#include "pmacc/types.hpp"

#include <mpi.h>

namespace pmacc
{
    namespace mpi
    {
        namespace reduceMethods
        {
            struct AllReduce
            {
                HINLINE bool hasResult(int mpiRank) const
                {
                    return mpiRank != -1;
                }

                template<class Functor, typename Type>
                HINLINE void operator()(
                    Functor,
                    Type* dest,
                    Type* src,
                    size_t const count,
                    MPI_Datatype type,
                    MPI_Op op,
                    MPI_Comm comm) const
                {
                    // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                    eventSystem::getTransactionEvent().waitForFinished();

                    MPI_CHECK(MPI_Allreduce((void*) src, (void*) dest, static_cast<int>(count), type, op, comm));
                }
            };

        } /*namespace reduceMethods*/

    } /*namespace mpi*/

} /*namespace pmacc*/
