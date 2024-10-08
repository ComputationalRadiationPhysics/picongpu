/* Copyright 2016-2024 Alexander Grund, Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/algorithms/reverseBits.hpp"
#include "pmacc/kernel/atomic.hpp"
#include "pmacc/lockstep/Kernel.hpp"
#include "pmacc/memory/buffers/HostDeviceBuffer.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    struct IdGenerator
    {
        template<typename T_Worker>
        HDINLINE uint64_t fetchInc(T_Worker const& worker)
        {
            return kernel::atomicAllInc(worker, nextId, ::alpaka::hierarchy::Grids());
        }

        template<typename T_Worker>
        HDINLINE uint64_t fetch(T_Worker const& worker)
        {
            return alpaka::atomicCas(worker.getAcc(), nextId, uint64_t{0u}, uint64_t{0u});
        }

        uint64_t* nextId;
    };


    class IdProvider : public ISimulationData
    {
    public:
        struct State
        {
            /** Next id to be returned */
            uint64_t nextId;
            /** First id used */
            uint64_t startId;
            /** Maximum number of processes ever used (never decreases) */
            uint64_t maxNumProc;
        };

        void synchronize() override
        {
            idBuffer.deviceToHost();
        };

        SimulationDataId getUniqueId() override
        {
            return name;
        }

        auto getDeviceGenerator()
        {
            return IdGenerator{idBuffer.getDeviceBuffer().data()};
        }

        /** Returns the state (e.g. for saving)
         *  Result is the same as the parameter to @ref setState
         */
        State getState()
        {
            idBuffer.deviceToHost();
            return State{*idBuffer.getHostBuffer().data(), m_startId, m_maxNumProc};
        }

        uint64_t getNewIdHost()
        {
            HostDeviceBuffer<uint64_t, 1> newIdBuf(DataSpace<1>(1));

            auto kernel = [] ALPAKA_FN_ACC(auto const& worker, auto idGenerator, uint64_t* nextId) -> void
            { *nextId = idGenerator.fetchInc(worker); };
            PMACC_LOCKSTEP_KERNEL(kernel).config<1>(1)(getDeviceGenerator(), newIdBuf.getDeviceBuffer().data());
            newIdBuf.deviceToHost();
            return *newIdBuf.getHostBuffer().data();
        }

        /** Sets the internal state (e.g. after a restart)
         */
        void setState(State const& state)
        {
            *idBuffer.getHostBuffer().data() = state.nextId;
            idBuffer.hostToDevice();
            m_startId = state.startId;
            if(m_maxNumProc < state.maxNumProc)
                m_maxNumProc = state.maxNumProc;
            log<ggLog::INFO>("(Re-)Initialized IdProvider with id=%1%/%2% and maxNumProc=%3%/%4%") % state.nextId
                % state.startId % state.maxNumProc % m_maxNumProc;
        }

        IdProvider(SimulationDataId providerName, uint64_t mpiRank, uint64_t numMpiRanks)
            : name(providerName)
            , idBuffer(DataSpace<1>{1})
        {
            auto startId = reverseBits(mpiRank);
            State state{startId, startId, numMpiRanks};
            setState(state);
        }

        /**
         * Return true, if an overflow of the counter is detected and hence there might be duplicate ids
         */
        bool isOverflown()
        {
            State curState = getState();
            /* Overflow happens, when an id puts bits into the bits used for ensuring uniqueness.
             * This are the n upper bits with n = highest bit set in the maximum id (which is maxNumProc_ - 1)
             * when counting the bits from 1 = right most bit
             * So first we calculate n, then remove the lowest bits of the next id so we have only the n upper bits
             * If any of them is non-zero, it is an overflow and we can have duplicate ids.
             * If not, then all ids are probably unique (still a chance, the id is overflown so much, that detection is
             * impossible)
             */
            uint64_t tmp = curState.maxNumProc - 1;
            int32_t bitsToCheck = 0;
            while(tmp)
            {
                bitsToCheck++;
                tmp >>= 1;
            }

            // Number of bits in the ids
            static constexpr int32_t numBitsOfType = sizeof(curState.maxNumProc) * CHAR_BIT;

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

    private:
        uint64_t m_maxNumProc = 0;
        uint64_t m_startId = 0;
        SimulationDataId name;

        pmacc::HostDeviceBuffer<uint64_t, 1> idBuffer;
    };

} // namespace pmacc
