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
    uint64_t IdProvider<T_dim>::maxNumProc_;

    template<unsigned T_dim>
    void IdProvider<T_dim>::init()
    {
        const uint64_t globalUniqueStartId = getStartId();
        maxNumProc_ = Environment<T_dim>::get().GridController().getGpuNodes().productOfComponents();
        setState(globalUniqueStartId, maxNumProc_);
        // Instantiate kernel
        getNewId();
        // Reset to start value
        setState(globalUniqueStartId, maxNumProc_);
    }

    template<unsigned T_dim>
    void IdProvider<T_dim>::setState(const uint64_t nextId, const uint64_t maxNumProc)
    {
        __cudaKernel(idDetail::setNextId)(1, 1)(nextId);
        if(maxNumProc_ < maxNumProc)
            maxNumProc_ = maxNumProc;
    }

    template<unsigned T_dim>
    boost::tuple<uint64_t, uint64_t> IdProvider<T_dim>::getState()
    {
        HostDeviceBuffer<uint64_cu, 1> nextIdBuf(DataSpace<1>(1));
        __cudaKernel(idDetail::getNextId)(1, 1)(nextIdBuf.getDeviceBuffer().getDataBox());
        nextIdBuf.deviceToHost();
        return boost::make_tuple(nextIdBuf.getHostBuffer().getDataBox()(0), maxNumProc_);
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
        /* Overflow happens, when an id puts bits into the bits used for ensuring uniqueness.
         * This are the n upper bits with n = highest bit set in maxNumProc_ (start counting at 1)
         * So first we calculate n, then remove the lowest bits of the next id so we have only the n upper bits
         * If any of them is non-zero, it is an overflow and we can have duplicate ids.
         * If not, then all ids are probably unique (still a chance, the id is overflown so much, that detection is impossible)
         */
        uint64_t tmpMaxNumProc = maxNumProc_;
        int32_t bitsToCheck = 0;
        while(tmpMaxNumProc)
        {
            bitsToCheck++;
            tmpMaxNumProc >>= 1;
        }

        // Number of bits in the ids
        BOOST_STATIC_CONSTEXPR int32_t numBitsOfType = sizeof(maxNumProc_) * CHAR_BIT;

        // Get current value
        uint64_t nextId = getState().get<0>();
        /* Example: maxNumProc_ has 3 set bits (<8 ranks), 64bit value used
         * --> Shift by 61 bits
         * => 3 upper bits are left untouched (besides moving), rest is zero
         */
        nextId >>= numBitsOfType - bitsToCheck;

        return nextId != 0;
    }

    template<unsigned T_dim>
    uint64_t IdProvider<T_dim>::getStartId()
    {
        uint64_t rank = Environment<T_dim>::get().GridController().getGlobalRank();

        /* We put the rank into the upper bits to have the lower bits for counting up and still
         * getting unique numbers. Reversing the bits instead of shifting gives some more room
         * as the upper bits of the rank are often also zero
         * Note: Overflow detection will still return true for that case
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
