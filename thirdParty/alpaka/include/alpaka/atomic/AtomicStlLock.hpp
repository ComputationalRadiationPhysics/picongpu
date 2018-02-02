/**
* \file
* Copyright 2014-2016 Benjamin Worpitz, Rene Widera
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/atomic/Traits.hpp>

#include <mutex>
#include <array>

namespace alpaka
{
    namespace atomic
    {
        //#############################################################################
        //! The CPU threads accelerator atomic ops.
        //
        //  Atomics can be used in the grids, blocks and threads hierarchy levels.
        //  Atomics are not guaranteed to be save between devices.
        //
        // \tparam THashTableSize size of the hash table to allow concurrency between
        //                        atomics to different addresses
        template<size_t THashTableSize>
        class AtomicStlLock
        {
        public:
            template<
                typename TAtomic,
                typename TOp,
                typename T,
                typename THierarchy,
                typename TSfinae>
            friend struct atomic::traits::AtomicOp;

            static constexpr size_t nextPowerOf2(size_t const value, size_t const bit = 0u)
            {
                return value <= (static_cast<size_t>(1u) << bit) ?
                    (static_cast<size_t>(1u) << bit) : nextPowerOf2(value, bit + 1u);
            }

            //-----------------------------------------------------------------------------
            //! get a hash value of the pointer
            //
            // This is no perfect hash, there will be collisions if the size of pointer type
            // is not a power of two.
            template<typename TPtr>
            static size_t hash(TPtr const * const ptr)
            {
                size_t const ptrAddr = reinterpret_cast< size_t >( ptr );
                // using power of two for the next division will increase the performance
                constexpr size_t typeSizePowerOf2 = nextPowerOf2(sizeof(TPtr));
                // division removes the stride between indices
                return (ptrAddr / typeSizePowerOf2);
            }

            //-----------------------------------------------------------------------------
            AtomicStlLock() = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicStlLock(AtomicStlLock const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicStlLock(AtomicStlLock &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AtomicStlLock const &) -> AtomicStlLock & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AtomicStlLock &&) -> AtomicStlLock & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AtomicStlLock() = default;

            template<typename TPtr>
            std::mutex & getMutex(TPtr const * const ptr) const
            {
                //-----------------------------------------------------------------------------
                //! get the size of the hash table
                //
                // The size is at least 1 or THashTableSize rounded up to the next power of 2
                constexpr size_t hashTableSize = THashTableSize == 0u ? 1u : nextPowerOf2(THashTableSize);

                size_t const hashedAddr = hash(ptr) & (hashTableSize - 1u);
                static std::array<
                    std::mutex,
                    hashTableSize> m_mtxAtomic; //!< The mutex protecting access for an atomic operation.
                return m_mtxAtomic[hashedAddr];
            }
        };

        namespace traits
        {
            //#############################################################################
            //! The CPU threads accelerator atomic operation.
            template<
                typename TOp,
                typename T,
                typename THierarchy,
                size_t THashTableSize>
            struct AtomicOp<
                TOp,
                atomic::AtomicStlLock<THashTableSize>,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA static auto atomicOp(
                    atomic::AtomicStlLock<THashTableSize> const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    std::lock_guard<std::mutex> lock(atomic.getMutex(addr));
                    return TOp()(addr, value);
                }
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA static auto atomicOp(
                    atomic::AtomicStlLock<THashTableSize> const & atomic,
                    T * const addr,
                    T const & compare,
                    T const & value)
                -> T
                {
                    std::lock_guard<std::mutex> lock(atomic.getMutex(addr));
                    return TOp()(addr, compare, value);
                }
            };
        }
    }
}
