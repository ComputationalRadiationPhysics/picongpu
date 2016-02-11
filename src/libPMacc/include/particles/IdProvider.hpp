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

#include "types.h"
#include "Environment.hpp"
#include "eventSystem/EventSystem.hpp"
#include "algorithms/reverseBits.hpp"
#include "nvidia/atomic.hpp"
#include "memory/buffers/HostDeviceBuffer.hpp"

namespace PMacc {

    namespace IdDetail {

        __device__ uint64_cu nextId;

        __global__ void setNextId(uint64_cu id)
        {
            nextId = id;
        }

        template<class T_Box>
        __global__ void getNextId(T_Box box)
        {
            box(0) = nextId;
        }

    }  // namespace IdDetail

    /**
     * Provider for globally unique (even across ranks) ids
     * Implemented for use in static contexts
     */
    class IdProvider
    {
    public:
        /** Initializes the state so it is read for use
         */
        static void init()
        {
            const uint64_t globalUniqueStartId = getStartId();
            setNextId(globalUniqueStartId);
        }

        /** Sets the next id to a given value (e.g. after a restart)
         */
        static void setNextId(const uint64_t nextId)
        {
            __cudaKernel(IdDetail::setNextId)(1, 1)(nextId);
        }

        /** Returns the next id without changing it (e.g. for saving)
         */
        static uint64_t getNextId()
        {
            HostDeviceBuffer<uint64_cu, 1> nextIdBuf(DataSpace<1>(1));
            __cudaKernel(IdDetail::getNextId)(1, 1)(nextIdBuf.getDeviceBuffer().getDataBox());
            nextIdBuf.deviceToHost();
            return nextIdBuf.getHostBuffer().getDataBox()(0);
        }

        /** Functor that returns a new id */
        struct GetNewId
        {
            uint64_cu operator()() const
            {
                return getNewId();
            }
        };

        /** Function that returns a new id */
        HDINLINE static uint64_cu getNewId()
        {
#ifdef __CUDA_ARCH__
            return nvidia::atomicAllInc(&IdDetail::nextId);
#else
            throw std::runtime_error("getNewId not implemented for host");
#endif
        }

        /**
         * Return true, if an overflow of the counter is detected and hence there might be duplicate ids
         */
        static bool isOverflown()
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

    private:
        static uint64_t getStartId()
        {
            uint64_t rank = Environment<>::get().GridController().getGlobalRank();

            /* We put the rank into the upper bits to have the lower bits for counting up and still
             * getting unique numbers. Reversing the bits instead of shifting gives some more room
             * as the upper bits of the rank or often also zero
             */
            return reverseBits(rank);
        }
    };

}  // namespace PMacc
