/* Copyright 2013-2021 Felix Schmitt, Rene Widera
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

#include <cstring> // memset

namespace pmacc
{
    namespace nvidia
    {
        namespace memory
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
                void getMemoryInfo(size_t* free, size_t* total = nullptr)
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

                /** Returns true if the memory pool is shared by host and device */
                bool isSharedMemoryPool()
                {
#if(PMACC_CUDA_ENABLED != 1 && ALPAKA_ACC_GPU_HIP_ENABLED != 1)
                    return true;
#else
                    size_t freeInternal = 0;
                    size_t freeAtStart = 0;

                    getMemoryInfo(&freeAtStart);

                    /* alloc 90%, since allocating 100% is a bit risky on a SoC-like device */
                    size_t allocSth = size_t(0.9 * double(freeAtStart));
                    uint8_t* c = new uint8_t[allocSth];
                    memset(c, 0, allocSth);

                    getMemoryInfo(&freeInternal);
                    delete[] c;

                    /* if we allocated 90% of available mem, we should have "lost" more
                     * than 50% of memory, even with fluctuations from the OS */
                    if(double(freeInternal) / double(freeAtStart) < 0.5)
                        return true;

                    return false;
#endif
                }

                void setReservedMemory(size_t reservedMem)
                {
                    this->reservedMem = reservedMem;
                }

            protected:
                size_t reservedMem;

            private:
                friend struct detail::Environment;

                static MemoryInfo& getInstance()
                {
                    static MemoryInfo instance;
                    return instance;
                }

                MemoryInfo() : reservedMem(0)
                {
                }
            };
        } // namespace memory
    } // namespace nvidia
} // namespace pmacc
