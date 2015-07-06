/**
 * Copyright 2013-2015 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "communication/manager_common.h"

#include "mpi/reduceMethods/AllReduce.hpp"
#include "mpi/GetMPI_StructAsArray.hpp"
#include "mpi/GetMPI_Op.hpp"

#include "types.h"

#include <cassert>

#include <mpi.h>

namespace PMacc
{
namespace mpi
{

struct MPIReduce
{

    /*reduce data over selected mpi nodes*/
    MPIReduce() : mpiRank(-1), numRanks(0), comm(MPI_COMM_NULL), isMPICommInitialized(false)
    {
        participate(true);
    }

    virtual ~MPIReduce()
    {
        if (isMPICommInitialized)
        {
            MPI_CHECK(MPI_Comm_free(&comm));
        }
    }

    /*
     * @return if resut of operator() is valid*/
    template<class MPIMethod>
    bool hasResult(const MPIMethod & method) const
    {
        assert(isMPICommInitialized == true);
        return method.hasResult(mpiRank);
    }

    /*
     * @return if resut of operator() is valid*/

    bool hasResult() const
    {
        return this->hasResult(::PMacc::mpi::reduceMethods::AllReduce());
    }

    /* Activate participation for reduce algorithm.
     * Must called from any mpi process. This function use global blocking mpi calls.
     * @param isActive true if mpi rank should be part of reduce operation, else false
     */
    void participate(bool isActive)
    {
        /*free old communicator of init is called again*/
        if (isMPICommInitialized)
        {
            MPI_CHECK(MPI_Comm_free(&comm));
            mpiRank = -1;
            numRanks = 0;
            isMPICommInitialized = false;
        }

        int countRanks;
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &countRanks));
        std::vector<int> reduceRank(countRanks);
        std::vector<int> groupRanks(countRanks);
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));

        if (!isActive)
            mpiRank = -1;

        MPI_CHECK(MPI_Allgather(&mpiRank, 1, MPI_INT, &reduceRank[0], 1, MPI_INT, MPI_COMM_WORLD));

        for (int i = 0; i < countRanks; ++i)
        {
            if (reduceRank[i] != -1)
            {
                groupRanks[numRanks] = reduceRank[i];
                numRanks++ ;
            }
        }

        MPI_Group group = MPI_GROUP_NULL;
        MPI_Group newgroup = MPI_GROUP_NULL;
        MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &group));
        MPI_CHECK(MPI_Group_incl(group, numRanks, &groupRanks[0], &newgroup));

        MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, newgroup, &comm));

        if (mpiRank != -1)
        {
            MPI_CHECK(MPI_Comm_rank(comm, &mpiRank));
            isMPICommInitialized = true;
        }
        MPI_CHECK(MPI_Group_free(&group));
        MPI_CHECK(MPI_Group_free(&newgroup));
    }

    /* Reduce elements on cpu memory
     * call hasResult to see if returned value is valid
     *
     * @param func binary functor for reduce which takes two arguments, first argument is the source and get the new reduced value.
     * Functor must specialize the function getMPI_Op.
     * @param dest buffer for result data
     * @param src a class or a pointer where the reduce algorithm can access the value by operator [] (one dimension access)
     * @param n number of elements to reduce
     * @param method mpi method for reduce
     *
     */
    template<class Functor, typename Type, class ReduceMethod >
    HINLINE void operator()(Functor func,
                             Type* dest,
                             Type* src,
                             const size_t n,
                             const ReduceMethod method)
    {
        typedef Type ValueType;

        method(func,
               dest,
               src,
               n * ::PMacc::mpi::getMPI_StructAsArray<ValueType > ().sizeMultiplier,
               ::PMacc::mpi::getMPI_StructAsArray<ValueType > ().dataType,
               ::PMacc::mpi::getMPI_Op<Functor > (),
               comm);
    }

    /* Reduce elements on cpu memory
     * the default reduce method is allReduce which means that any host get the reduced value back
     *
     * @param func binary functor for reduce which takes two arguments, first argument is the source and get the new reduced value.
     * Functor must specialize the function getMPI_Op.
     * @param dest buffer for result data
     * @param src a class or a pointer where the reduce algorithm can access the value by operator [] (one dimension access)
     * @param n number of elements to reduce
     *
     * @return reduced value
     */
    template<class Functor, typename Type >
    HINLINE void operator()(Functor func,
                             Type* dest,
                             Type* src,
                             const size_t n)
    {
        this->operator ()(func, dest, src, n, ::PMacc::mpi::reduceMethods::AllReduce());
    }


private:

    MPI_Comm comm;
    int mpiRank;
    int numRanks;
    bool isMPICommInitialized;
};
}
}//namespace

