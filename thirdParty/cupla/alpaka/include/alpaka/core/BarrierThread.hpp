/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Uncomment this to disable the standard spinlock behaviour of the threads
// #define ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK

#include "alpaka/block/sync/Traits.hpp"
#include "alpaka/core/Common.hpp"

#include <condition_variable>
#include <mutex>
#ifndef ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK
#    include <atomic>
#    include <thread>
#endif

namespace alpaka::core
{
    namespace threads
    {
        //! A self-resetting barrier.
        template<typename TIdx>
        class BarrierThread final
        {
        public:
            explicit BarrierThread(TIdx const& threadCount)
                : m_threadCount(threadCount)
                , m_curThreadCount(threadCount)
                , m_generation(0)
            {
            }

            //! Waits for all the other threads to reach the barrier.
            auto wait() -> void
            {
                TIdx const generationWhenEnteredTheWait = m_generation;
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
                    m_cvAllThreadsReachedBarrier.wait(
                        lock,
                        [this, generationWhenEnteredTheWait] { return generationWhenEnteredTheWait != m_generation; });
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
            const TIdx m_threadCount;
#ifdef ALPAKA_THREAD_BARRIER_DISABLE_SPINLOCK
            TIdx m_curThreadCount;
            TIdx m_generation;
#else
            std::atomic<TIdx> m_curThreadCount;
            std::atomic<TIdx> m_generation;
#endif
        };

        namespace detail
        {
            template<typename TOp>
            struct AtomicOp;

            template<>
            struct AtomicOp<BlockCount>
            {
                void operator()(std::atomic<int>& result, bool value)
                {
                    result += static_cast<int>(value);
                }
            };

            template<>
            struct AtomicOp<BlockAnd>
            {
                void operator()(std::atomic<int>& result, bool value)
                {
                    result &= static_cast<int>(value);
                }
            };

            template<>
            struct AtomicOp<BlockOr>
            {
                void operator()(std::atomic<int>& result, bool value)
                {
                    result |= static_cast<int>(value);
                }
            };
        } // namespace detail

        //! A self-resetting barrier with barrier.
        template<typename TIdx>
        class BarrierThreadWithPredicate final
        {
        public:
            explicit BarrierThreadWithPredicate(TIdx const& threadCount)
                : m_threadCount(threadCount)
                , m_curThreadCount(threadCount)
                , m_generation(0)
            {
            }

            //! Waits for all the other threads to reach the barrier.
            template<typename TOp>
            ALPAKA_FN_HOST auto wait(int predicate) -> int
            {
                TIdx const generationWhenEnteredTheWait = m_generation;
                std::unique_lock<std::mutex> lock(m_mtxBarrier);

                auto const generationMod2 = m_generation % static_cast<TIdx>(2u);
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
                    m_cvAllThreadsReachedBarrier.wait(
                        lock,
                        [this, generationWhenEnteredTheWait] { return generationWhenEnteredTheWait != m_generation; });
                }
                return m_result[generationMod2];
            }

        private:
            std::mutex m_mtxBarrier;
            std::condition_variable m_cvAllThreadsReachedBarrier;
            const TIdx m_threadCount;
            TIdx m_curThreadCount;
            TIdx m_generation;
            std::atomic<int> m_result[2];
        };
    } // namespace threads
} // namespace alpaka::core
