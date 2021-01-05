/* Copyright 2013-2021 Heiko Burau, Rene Widera
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

#include "pmacc/types.hpp"
#include <mpi.h>
#include "pmacc/communication/manager_common.hpp"

namespace pmacc
{
    namespace mpi
    {
        namespace reduceMethods
        {
            struct Reduce
            {
                HINLINE bool hasResult(int mpiRank) const
                {
                    return mpiRank == 0;
                }

                template<class Functor, typename Type>
                HINLINE void operator()(
                    Functor,
                    Type* dest,
                    Type* src,
                    const size_t count,
                    MPI_Datatype type,
                    MPI_Op op,
                    MPI_Comm comm) const
                {
                    // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                    __getTransactionEvent().waitForFinished();

                    MPI_CHECK(MPI_Reduce((void*) src, (void*) dest, count, type, op, 0, comm));
                }
            };

        } /*namespace reduceMethods*/

    } /*namespace mpi*/

} /*namespace pmacc*/
