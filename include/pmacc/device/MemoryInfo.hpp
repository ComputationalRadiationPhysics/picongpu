/* Copyright 2013-2023 Felix Schmitt, Rene Widera
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


#include "pmacc/Environment.def"
#include "pmacc/types.hpp"

#include <pmacc/communication/manager_common.hpp>

#include <cstring> // memset

#include <mpi.h>

namespace pmacc
{
    namespace device
    {
        /**
         * Provides convenience methods for querying memory information.
         * Singleton class.
         */
        class MemoryInfo
        {
        public:
            /**
             * Returns information about device memory.
             *
             * @param free amount of free memory in bytes. can be nullptr
             * @param total total amount of memory in bytes. can be nullptr. (nullptr by default)
             */
            void getMemoryInfo(size_t* free, size_t* total = nullptr) const
            {
                size_t freeInternal = 0;
                size_t totalInternal = 0;

                CUDA_CHECK(cuplaMemGetInfo(&freeInternal, &totalInternal));

                if(free != nullptr)
                {
                    if(reservedMem > freeInternal)
                        freeInternal = 0;
                    else
                        freeInternal -= reservedMem;

                    *free = freeInternal;
                }
                if(total != nullptr)
                {
                    if(reservedMem > totalInternal)
                        totalInternal = 0;
                    else
                        totalInternal -= reservedMem;

                    *total = totalInternal;
                }
            }

            /** Check if the memory pool is shared by host and device.
             *
             * @attention This method is using MPI collectives and must be called from all MPI processes collectively.
             *
             *  @param numRanksPerDevice number of ranks using one device
             *  @param mpiComm MPI communicator
             *
             *  @return true if the device memory is shared with the host else false
             */
            bool isSharedMemoryPool(
                [[maybe_unused]] uint32_t const numRanksPerDevice,
                [[maybe_unused]] MPI_Comm mpiComm) const
            {
#if(PMACC_CUDA_ENABLED != 1 && ALPAKA_ACC_GPU_HIP_ENABLED != 1)
                return true;
#else
                if(numRanksPerDevice >= 2u)
                    MPI_CHECK(MPI_Barrier(mpiComm));

                size_t freeInternal = 0;
                size_t freeAtStart = 0;

                getMemoryInfo(&freeAtStart);

                if(numRanksPerDevice >= 2u)
                    MPI_CHECK(MPI_Barrier(mpiComm));

                /* Do not allocate 100% is a bit risky on a SoC-like device */
                double const fractionOfMemory = 0.9 / static_cast<double>(numRanksPerDevice);
                size_t allocSth = static_cast<size_t>(fractionOfMemory * static_cast<double>(freeAtStart));
                uint8_t* c = new uint8_t[allocSth];
                memset(c, 0, allocSth);

                if(numRanksPerDevice >= 2u)
                    MPI_CHECK(MPI_Barrier(mpiComm));

                getMemoryInfo(&freeInternal);

                if(numRanksPerDevice >= 2u)
                    MPI_CHECK(MPI_Barrier(mpiComm));

                delete[] c;

                /* if we allocated 90% of available mem, we should have "lost" more
                 * than 50% of memory, even with fluctuations from the OS */
                double const thresholdFraction = 0.5 / static_cast<double>(numRanksPerDevice);
                if(static_cast<double>(freeInternal) / static_cast<double>(freeAtStart) < thresholdFraction)
                    return true;

                return false;
#endif
            }

            void setReservedMemory(size_t reservedMem)
            {
                this->reservedMem = reservedMem;
            }

        protected:
            size_t reservedMem{0};

        private:
            friend struct detail::Environment;

            static MemoryInfo& getInstance()
            {
                static MemoryInfo instance;
                return instance;
            }

            MemoryInfo() = default;
        };

    } // namespace device
} // namespace pmacc
