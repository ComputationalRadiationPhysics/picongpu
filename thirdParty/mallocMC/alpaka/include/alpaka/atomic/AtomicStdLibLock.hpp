/* Copyright 2019 Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/atomic/Traits.hpp>

#include <alpaka/core/BoostPredef.hpp>

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
        class AtomicStdLibLock
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
            AtomicStdLibLock() = default;
            //-----------------------------------------------------------------------------
            AtomicStdLibLock(AtomicStdLibLock const &) = delete;
            //-----------------------------------------------------------------------------
            AtomicStdLibLock(AtomicStdLibLock &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(AtomicStdLibLock const &) -> AtomicStdLibLock & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(AtomicStdLibLock &&) -> AtomicStdLibLock & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AtomicStdLibLock() = default;

            template<typename TPtr>
            std::mutex & getMutex(TPtr const * const ptr) const
            {
                //-----------------------------------------------------------------------------
                //! get the size of the hash table
                //
                // The size is at least 1 or THashTableSize rounded up to the next power of 2
                constexpr size_t hashTableSize = THashTableSize == 0u ? 1u : nextPowerOf2(THashTableSize);

                size_t const hashedAddr = hash(ptr) & (hashTableSize - 1u);
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif
                static std::array<
                    std::mutex,
                    hashTableSize> m_mtxAtomic; //!< The mutex protecting access for an atomic operation.
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
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
                atomic::AtomicStdLibLock<THashTableSize>,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicStdLibLock<THashTableSize> const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    std::lock_guard<std::mutex> lock(atomic.getMutex(addr));
                    return TOp()(addr, value);
                }
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicStdLibLock<THashTableSize> const & atomic,
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
