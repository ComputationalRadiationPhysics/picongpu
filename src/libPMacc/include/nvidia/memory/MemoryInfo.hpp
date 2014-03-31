/**
 * Copyright 2013 Felix Schmitt, Rene Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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
#include "types.h"

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

    void setReservedMemory(size_t reservedMem)
    {
        this->reservedMem = reservedMem;
    }

protected:
    size_t reservedMem;

private:
    
    friend Environment<DIM1>;
    friend Environment<DIM2>;
    friend Environment<DIM3>;
    
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


