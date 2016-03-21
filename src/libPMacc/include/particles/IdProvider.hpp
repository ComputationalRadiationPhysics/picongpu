/**
 * Copyright 2016 Alexander Grund
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

#include "IdProvider.def"
#include "Environment.hpp"
#include "eventSystem/EventSystem.hpp"
#include "algorithms/reverseBits.hpp"
#include "nvidia/atomic.hpp"
#include "memory/buffers/HostDeviceBuffer.hpp"

namespace PMacc {

    namespace idDetail {

        __device__ uint64_cu nextId;

        __global__ void setNextId(uint64_cu id)
        {
            nextId = id;
        }

        template<class T_Box>
        __global__ void getNextId(T_Box boxOut)
        {
            boxOut(0) = nextId;
        }

        template<class T_Box, class T_GetNewId>
        __global__ void getNewId(T_Box boxOut, T_GetNewId getNewId)
        {
            boxOut(0) = getNewId();
        }

    }  // namespace idDetail

    template<unsigned T_dim>
    void IdProvider<T_dim>::init()
    {
        const uint64_t globalUniqueStartId = getStartId();
        setNextId(globalUniqueStartId);
        // Instantiate kernel
        getNewIdHost();
        // Reset to start value
        setNextId(globalUniqueStartId);
    }

    template<unsigned T_dim>
    void IdProvider<T_dim>::setNextId(const uint64_t nextId)
    {
        __cudaKernel(idDetail::setNextId)(1, 1)(nextId);
    }

    template<unsigned T_dim>
    uint64_t IdProvider<T_dim>::getNextId()
    {
        HostDeviceBuffer<uint64_cu, 1> nextIdBuf(DataSpace<1>(1));
        __cudaKernel(idDetail::getNextId)(1, 1)(nextIdBuf.getDeviceBuffer().getDataBox());
        nextIdBuf.deviceToHost();
        return nextIdBuf.getHostBuffer().getDataBox()(0);
    }

    template<unsigned T_dim>
    HDINLINE uint64_cu IdProvider<T_dim>::getNewId()
    {
#ifdef __CUDA_ARCH__
        return nvidia::atomicAllInc(&idDetail::nextId);
#else
        // IMPORTANT: This calls a kernel. So make sure this kernel is instantiated somewhere before!
        return getNewIdHost();
#endif
    }

    template<unsigned T_dim>
    bool IdProvider<T_dim>::isOverflown()
    {
        // Get current value
        uint64_t nextId = getNextId();
        // Get start value
        uint64_t globalUniqueStartId = getStartId();

        /* Start value contains globally unique bits in the high order bits
         * Hence compare all high order bits till the last set one and check,
         * if any of them differs from the start value --> overflow
         */
        // How far do we need to shift to get the highest bit
        BOOST_STATIC_CONSTEXPR int bitsToHighest = sizeof(uint64_t) * CHAR_BIT - 1;
        while(globalUniqueStartId)
        {
            // Compare highest bit
            if(globalUniqueStartId >> bitsToHighest != nextId >> bitsToHighest)
                return true;
            // Proceed to next bit
            globalUniqueStartId <<= 1;
        }
        return false;
    }

    template<unsigned T_dim>
    uint64_t IdProvider<T_dim>::getStartId()
    {
        uint64_t rank = Environment<T_dim>::get().GridController().getGlobalRank();

        /* We put the rank into the upper bits to have the lower bits for counting up and still
         * getting unique numbers. Reversing the bits instead of shifting gives some more room
         * as the upper bits of the rank are often also zero
         */
        return reverseBits(rank);
    }

    template<unsigned T_dim>
    uint64_t IdProvider<T_dim>::getNewIdHost()
    {
        HostDeviceBuffer<uint64_cu, 1> newIdBuf(DataSpace<1>(1));
        __cudaKernel(idDetail::getNewId)(1, 1)(newIdBuf.getDeviceBuffer().getDataBox(), GetNewId());
        newIdBuf.deviceToHost();
        return newIdBuf.getHostBuffer().getDataBox()(0);
    }

}  // namespace PMacc
