/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

// Uncomment this to disable the standard spinlock behaviour of the threads
//#define ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK

#include <alpaka/core/Common.hpp>
#include <alpaka/block/sync/Traits.hpp>

#include <mutex>
#include <condition_variable>
#ifndef ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK
    #include <atomic>
    #include <thread>
#endif

namespace alpaka
{
    namespace core
    {
        namespace threads
        {
            //#############################################################################
            //! A self-resetting barrier.
            template<
                typename TSize>
            class BarrierThread final
            {
            public:
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA explicit BarrierThread(
                    TSize const & threadCount) :
                    m_threadCount(threadCount),
                    m_curThreadCount(threadCount),
                    m_generation(0)
                {}
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BarrierThread(BarrierThread const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BarrierThread(BarrierThread &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BarrierThread const &) -> BarrierThread & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BarrierThread &&) -> BarrierThread & = delete;
                //-----------------------------------------------------------------------------
                ~BarrierThread() = default;

                //-----------------------------------------------------------------------------
                //! Waits for all the other threads to reach the barrier.
                ALPAKA_FN_ACC_NO_CUDA auto wait()
                -> void
                {
                    TSize const generationWhenEnteredTheWait = m_generation;
#ifdef ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK
                    std::unique_lock<std::mutex> lock(m_mtxBarrier);
#endif
                    if(--m_curThreadCount == 0)
                    {
                        m_curThreadCount = m_threadCount;
                        ++m_generation;
#ifdef ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK
                        m_cvAllThreadsReachedBarrier.notify_all();
#endif
                    }
                    else
                    {
#ifdef ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK
                        m_cvAllThreadsReachedBarrier.wait(lock, [this, generationWhenEnteredTheWait] { return generationWhenEnteredTheWait != m_generation; });
#else
                        while(generationWhenEnteredTheWait == m_generation)
                        {
                            std::this_thread::yield();
                        }
#endif
                    }
                }

            private:
#ifdef ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK
                std::mutex m_mtxBarrier;
                std::condition_variable m_cvAllThreadsReachedBarrier;
#endif
                const TSize m_threadCount;
#ifdef ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK
                TSize m_curThreadCount;
                TSize m_generation;
#else
                std::atomic<TSize> m_curThreadCount;
                std::atomic<TSize> m_generation;
#endif
            };

            namespace detail
            {
                //#############################################################################
                template<
                    typename TOp>
                struct AtomicOp;
                //#############################################################################
                template<>
                struct AtomicOp<
                    block::sync::op::Count>
                {
                    void operator()(std::atomic<int>& result, bool value)
                    {
                        result += static_cast<int>(value);
                    }
                };
                //#############################################################################
                template<>
                struct AtomicOp<
                    block::sync::op::LogicalAnd>
                {
                    void operator()(std::atomic<int>& result, bool value)
                    {
                        result &= static_cast<int>(value);
                    }
                };
                //#############################################################################
                template<>
                struct AtomicOp<
                    block::sync::op::LogicalOr>
                {
                    void operator()(std::atomic<int>& result, bool value)
                    {
                        result |= static_cast<int>(value);
                    }
                };
            }

            //#############################################################################
            //! A self-resetting barrier with barrier.
            template<
                typename TSize>
            class BarrierThreadWithPredicate final
            {
            public:
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA explicit BarrierThreadWithPredicate(
                    TSize const & threadCount) :
                    m_threadCount(threadCount),
                    m_curThreadCount(threadCount),
                    m_generation(0)
                {}
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BarrierThreadWithPredicate(BarrierThreadWithPredicate const & other) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA BarrierThreadWithPredicate(BarrierThreadWithPredicate &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BarrierThreadWithPredicate const &) -> BarrierThreadWithPredicate & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(BarrierThreadWithPredicate &&) -> BarrierThreadWithPredicate & = delete;
                //-----------------------------------------------------------------------------
                ~BarrierThreadWithPredicate() = default;

                //-----------------------------------------------------------------------------
                //! Waits for all the other threads to reach the barrier.
                template<
                    typename TOp>
                ALPAKA_FN_ACC_NO_CUDA auto wait(int predicate)
                -> int
                {
                    TSize const generationWhenEnteredTheWait = m_generation;
                    std::unique_lock<std::mutex> lock(m_mtxBarrier);

                    auto const generationMod2(m_generation % static_cast<TSize>(2u));
                    if(m_curThreadCount == m_threadCount)
                    {
                        m_result[generationMod2] = TOp::InitialValue;
                    }

                    std::atomic<int>& result(m_result[generationMod2]);
                    bool const predicateBool(predicate != 0);

                    detail::AtomicOp<TOp>()(result, predicateBool);

                    if(--m_curThreadCount == 0)
                    {
                        m_curThreadCount = m_threadCount;
                        ++m_generation;
                        m_cvAllThreadsReachedBarrier.notify_all();
                    }
                    else
                    {
                        m_cvAllThreadsReachedBarrier.wait(lock, [this, generationWhenEnteredTheWait] { return generationWhenEnteredTheWait != m_generation; });
                    }
                    return m_result[generationMod2];
                }

            private:
                std::mutex m_mtxBarrier;
                std::condition_variable m_cvAllThreadsReachedBarrier;
                const TSize m_threadCount;
                TSize m_curThreadCount;
                TSize m_generation;
                std::atomic<int> m_result[2];
            };
        }
    }
}
