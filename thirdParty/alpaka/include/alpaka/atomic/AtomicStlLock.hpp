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

#include <alpaka/atomic/Traits.hpp>                 // AtomicOp

#include <mutex>                                    // std::mutex, std::lock_guard

namespace alpaka
{
    namespace atomic
    {
        //#############################################################################
        //! The CPU threads accelerator atomic ops.
        //
        //  Atomics can be used in the grids, blocks and threads hierarchy levels.
        //  Atomics are not guaranteed to be save between devices.
        //#############################################################################
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

            //-----------------------------------------------------------------------------
            //! Default constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicStlLock() = default;
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicStlLock(AtomicStlLock const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicStlLock(AtomicStlLock &&) = delete;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AtomicStlLock const &) -> AtomicStlLock & = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AtomicStlLock &&) -> AtomicStlLock & = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~AtomicStlLock() = default;

            std::mutex & getMutex() const
            {
                static std::mutex m_mtxAtomic; //!< The mutex protecting access for a atomic operation.
                return m_mtxAtomic;
            }
        };

        namespace traits
        {
            //#############################################################################
            //! The CPU threads accelerator atomic operation.
            //#############################################################################
            template<
                typename TOp,
                typename T,
                typename THierarchy>
            struct AtomicOp<
                TOp,
                atomic::AtomicStlLock,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA static auto atomicOp(
                    atomic::AtomicStlLock const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // \TODO: Currently not only the access to the same memory location is protected by a mutex but all atomic ops on all threads.
                    // We could use a list of mutexes and lock the mutex depending on the target memory location to allow multiple atomic ops on different targets concurrently.
                    std::lock_guard<std::mutex> lock(atomic.getMutex());
                    return TOp()(addr, value);
                }
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA static auto atomicOp(
                    atomic::AtomicStlLock const & atomic,
                    T * const addr,
                    T const & compare,
                    T const & value)
                -> T
                {
                    // \TODO: Currently not only the access to the same memory location is protected by a mutex but all atomic ops on all threads.
                    // We could use a list of mutexes and lock the mutex depending on the target memory location to allow multiple atomic ops on different targets concurrently.
                    std::lock_guard<std::mutex> lock(atomic.getMutex());
                    return TOp()(addr, compare, value);
                }
            };
        }
    }
}
