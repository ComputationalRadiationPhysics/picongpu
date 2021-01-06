/* Copyright 2013-2021 Axel Huebl, Rene Widera
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

#include "pmacc/nvidia/reduce/Reduce.hpp"
#include "pmacc/mpi/MPIReduce.hpp"
#include "pmacc/traits/GetValueType.hpp"

namespace pmacc
{
    namespace algorithms
    {
        /* Reduce values in GPU memory over all MPI instances
         */
        class GlobalReduce
        {
        public:
            GlobalReduce(const uint32_t byte, const uint32_t sharedMemByte = 4 * 1024) : reduce(byte, sharedMemByte)
            {
            }

            /* Activate participation for reduce algorithm.
             * Must called from any mpi process. This function use global blocking mpi calls.
             * Don't create a instance befor you have set you cuda device!
             * @param isActive true if mpi rank should be part of reduce operation, else false
             */
            void participate(bool isActive)
            {
                mpi_reduce.participate(isActive);
            }

            /* Reduce elements in global gpu memeory
             *
             * @param func functor for reduce which takes two arguments, first argument is the source and get the new
             * reduced value. Functor must specialize the function getMPI_Op.
             * @param src a class or a pointer where the reduce algorithm can access the value by operator [] (one
             * dimension access)
             * @param n number of elements to reduce
             *
             * @return reduced value (same on every mpi instance)
             */
            template<class Functor, typename Src>
            typename traits::GetValueType<Src>::ValueType operator()(Functor func, Src src, uint32_t n)
            {
                typedef typename traits::GetValueType<Src>::ValueType Type;

                Type localResult = reduce(func, src, n);
                Type globalResult;

                mpi_reduce(func, &globalResult, &localResult, 1);
                return globalResult;
            }

        private:
            ::pmacc::nvidia::reduce::Reduce reduce;
            ::pmacc::mpi::MPIReduce mpi_reduce;
        };
    } // namespace algorithms
} // namespace pmacc
