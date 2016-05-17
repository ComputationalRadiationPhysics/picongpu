/**
 * Copyright 2013-2016 Felix Schmitt, Rene Widera
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

#include <cuda.h>
#include "pmacc_types.hpp"

#include <cstring> // memset

namespace PMacc
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
     * @param free amount of free memory in bytes. can be NULL
     * @param total total amount of memory in bytes. can be NULL. (NULL by default)
     */
    void getMemoryInfo(size_t *free, size_t *total = NULL)
    {
        size_t freeInternal = 0;
        size_t totalInternal = 0;

        CUDA_CHECK(cudaMemGetInfo(&freeInternal, &totalInternal));

        if (free != NULL)
        {
            if (reservedMem > freeInternal)
                freeInternal = 0;
            else
                freeInternal -= reservedMem;

            *free = freeInternal;
        }
        if (total != NULL)
        {
            if (reservedMem > totalInternal)
                totalInternal = 0;
            else
                totalInternal -= reservedMem;

            *total = totalInternal;
        }
    }

    /** Returns true if the memory pool is shared by host and device */
    bool isSharedMemoryPool()
    {
        size_t freeInternal = 0;
        size_t freeAtStart = 0;

        getMemoryInfo(&freeAtStart);

        /* alloc 90%, since allocating 100% is a bit risky on a SoC-like device */
        size_t allocSth = size_t( 0.9 * double(freeAtStart) );
        uint8_t* c = new uint8_t[allocSth];
        memset(c, 0, allocSth);

        getMemoryInfo(&freeInternal);
        delete [] c;

        /* if we allocated 90% of available mem, we should have "lost" more
         * than 50% of memory, even with fluctuations from the OS */
        if( double(freeInternal)/double(freeAtStart) < 0.5 )
            return true;

        return false;
    }

    void setReservedMemory(size_t reservedMem)
    {
        this->reservedMem = reservedMem;
    }

protected:
    size_t reservedMem;

private:
    friend class Environment<DIM1>;
    friend class Environment<DIM2>;
    friend class Environment<DIM3>;

    static MemoryInfo& getInstance()
    {
        static MemoryInfo instance;
        return instance;
    }

    MemoryInfo() :
    reservedMem(0)
    {

    }
};
} //namespace memory
} //namespace nvidia
} //namespace PMacc


