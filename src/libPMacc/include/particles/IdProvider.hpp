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
#include "debug/PMaccVerbose.hpp"

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
    uint64_t IdProvider<T_dim>::m_maxNumProc;
    template<unsigned T_dim>
    uint64_t IdProvider<T_dim>::m_startId;

    template<unsigned T_dim>
    void IdProvider<T_dim>::init()
    {
        // Init static variables
        m_startId = m_maxNumProc = 0;

        State state;
        state.startId = state.nextId = calcStartId();
        // Init value to avoid uninitialized read warnings
        setNextId(state.startId);
        // Instantiate kernel (Omitting this will result in silent crashes at runtime)
        getNewIdHost();
        // Reset to start value
        state.maxNumProc = Environment<T_dim>::get().GridController().getGpuNodes().productOfComponents();
        setState(state);
    }

    template<unsigned T_dim>
    void IdProvider<T_dim>::setState(const State& state)
    {
        setNextId(state.nextId);
        m_startId = state.startId;
        if(m_maxNumProc < state.maxNumProc)
            m_maxNumProc = state.maxNumProc;
        log<ggLog::INFO>("(Re-)Initialized IdProvider with id=%1%/%2% and maxNumProc=%3%/%4%")
                % state.nextId % state.startId
                % state.maxNumProc % m_maxNumProc;
    }

    template<unsigned T_dim>
    IdProvider<T_dim>::State IdProvider<T_dim>::getState()
    {
        HostDeviceBuffer<uint64_cu, 1> nextIdBuf(DataSpace<1>(1));
        __cudaKernel(idDetail::getNextId)(1, 1)(nextIdBuf.getDeviceBuffer().getDataBox());
        nextIdBuf.deviceToHost();
        State state;
        state.nextId = static_cast<uint64_t>(nextIdBuf.getHostBuffer().getDataBox()(0));
        state.startId = m_startId;
        state.maxNumProc = m_maxNumProc;
        return state;
    }

    template<unsigned T_dim>
    HDINLINE uint64_t IdProvider<T_dim>::getNewId()
    {
#ifdef __CUDA_ARCH__
        return static_cast<uint64_t>(nvidia::atomicAllInc(&idDetail::nextId));
#else
        // IMPORTANT: This calls a kernel. So make sure this kernel is instantiated somewhere before!
        return getNewIdHost();
#endif
    }

    template<unsigned T_dim>
    bool IdProvider<T_dim>::isOverflown()
    {
        State curState = getState();
        /* Overflow happens, when an id puts bits into the bits used for ensuring uniqueness.
         * This are the n upper bits with n = highest bit set in the maximum id (which is maxNumProc_ - 1)
         * when counting the bits from 1 = right most bit
         * So first we calculate n, then remove the lowest bits of the next id so we have only the n upper bits
         * If any of them is non-zero, it is an overflow and we can have duplicate ids.
         * If not, then all ids are probably unique (still a chance, the id is overflown so much, that detection is impossible)
         */
        uint64_t tmp = curState.maxNumProc - 1;
        int32_t bitsToCheck = 0;
        while(tmp)
        {
            bitsToCheck++;
            tmp >>= 1;
        }

        // Number of bits in the ids
        BOOST_STATIC_CONSTEXPR int32_t numBitsOfType = sizeof(curState.maxNumProc) * CHAR_BIT;

        // Get current id
        uint64_t nextId = curState.nextId;
        // Cancel out start id via xor -> Upper n bits should be 0
        nextId ^= curState.startId;
        /* Prepare to compare only upper n bits for 0
         * Example: maxNumProc_ has 3 set bits (<8 ranks), 64bit value used
         * --> Shift by 61 bits
         * => 3 upper bits are left untouched (besides moving), rest is zero
         */
        nextId >>= numBitsOfType - bitsToCheck;

        return nextId != 0;
    }

    template<unsigned T_dim>
    uint64_t IdProvider<T_dim>::calcStartId()
    {
        uint64_t rank = Environment<T_dim>::get().GridController().getScalarPosition();

        /* We put the rank into the upper bits to have the lower bits for counting up and still
         * getting unique numbers. Reversing the bits instead of shifting gives some more room
         * as the upper bits of the rank are often also zero
         * Note: Overflow detection will still return true for that case
         */
        return reverseBits(rank);
    }

    template<unsigned T_dim>
    void IdProvider<T_dim>::setNextId(uint64_t nextId)
    {
        __cudaKernel(idDetail::setNextId)(1, 1)(nextId);
    }

    template<unsigned T_dim>
    uint64_t IdProvider<T_dim>::getNewIdHost()
    {
        HostDeviceBuffer<uint64_cu, 1> newIdBuf(DataSpace<1>(1));
        __cudaKernel(idDetail::getNewId)(1, 1)(newIdBuf.getDeviceBuffer().getDataBox(), GetNewId());
        newIdBuf.deviceToHost();
        return static_cast<uint64_t>(newIdBuf.getHostBuffer().getDataBox()(0));
    }

}  // namespace PMacc
