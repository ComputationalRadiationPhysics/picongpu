/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/device/Reduce.hpp"
#include "pmacc/mpi/MPIReduce.hpp"
#include "pmacc/traits/GetValueType.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace algorithms
    {
        /* Reduce values in GPU memory over all MPI instances
         */
        class GlobalReduce
        {
        public:
            /** Constructor
             *
             * @param byte how many bytes in global gpu memory can be reserved for the reduction algorithm
             * @param sharedMemByte limit the usage of shared memory per block on gpu
             */
            GlobalReduce(uint32_t const byte, uint32_t const sharedMemByte = 4 * 1024) : reduce(byte, sharedMemByte)
            {
            }

            /* Activate participation for reduce algorithm.
             *
             * @attention Must be called from any mpi process. This function uses global blocking mpi calls.
             *
             * @param isActive true if MPI rank should participate in the reduction, else false
             */
            void participate(bool isActive)
            {
                isParticipating = isActive;
                mpi_reduce.participate(isActive);
            }

            /* defines if the result of the MPI operation is valid
             *
             * @tparam MPIMethod type of the reduction method
             * @param method used reduction method e.g.,
             *                reduceMethods::AllReduce, reduceMethods::Reduce
             * @return if result of operator() is valid*/
            template<class MPIMethod>
            bool hasResult(MPIMethod const& method = ::pmacc::mpi::reduceMethods::AllReduce())
            {
                return mpi_reduce.hasResult(method);
            }

            /* Reduce elements in global gpu memeory
             *
             * @param func Binary functor for the reduction. First parameter is used as input and result value. Functor
             * must specialize the function getMPI_Op.
             * @param src A class or a pointer where the reduction algorithm can access the value by operator [] (one
             * dimension access). The data must be located on the device.
             * @param n number of elements to reduce
             *
             * @return reduced value (same on every mpi instance)
             *
             * @{
             */
            template<class Functor, typename Src>
            typename traits::GetValueType<Src>::ValueType operator()(Functor func, Src src, uint32_t n)
            {
                return (*this)(func, src, n, ::pmacc::mpi::reduceMethods::AllReduce());
            }

            /**
             * @tparam MPIMethod type of the reduction method
             * @param method used reduction method e.g.,
             *               reduceMethods::AllReduce, reduceMethods::Reduce
             */
            template<class Functor, typename Src, class MPIMethod>
            typename traits::GetValueType<Src>::ValueType operator()(
                Functor func,
                Src src,
                uint32_t n,
                MPIMethod const& method)
            {
                using Type = typename traits::GetValueType<Src>::ValueType;

                if(isParticipating)
                {
                    Type localResult = reduce(func, src, n);
                    Type globalResult;

                    mpi_reduce(func, &globalResult, &localResult, 1, method);
                    return globalResult;
                }
                return Type{};
            }

            /** @} */

        private:
            /** cache if the rank is participating into the reduction */
            bool isParticipating = false;
            ::pmacc::device::Reduce reduce;
            ::pmacc::mpi::MPIReduce mpi_reduce;
        };
    } // namespace algorithms
} // namespace pmacc
