/* Copyright 2026 Mykhailo Varvarin, Maria Michailidi, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#    include "alpaka/core/Common.hpp"

#    include <oneapi/tbb/task.h>

#    include <mutex>
#    include <vector>

namespace alpaka::core
{
    namespace tbb
    {
        // A reusable barrier for TBB tasks using suspend/resume
        template<typename TIdx>
        class BarrierThread final
        {
        public:
            explicit BarrierThread(TIdx const& threadCount) : m_threadCount(threadCount)
            {
                assertValueUnsigned(threadCount);
                m_suspended.reserve(static_cast<std::size_t>(threadCount));
            }

            // Called from inside a task to wait until all have arrived
            auto wait() -> void
            {
                oneapi::tbb::task::suspend(
                    [this](oneapi::tbb::task::suspend_point sp)
                    {
                        // The last task to arrive takes all the suspended tasks, and leaves the barrier empty and
                        // ready to be reused by the next wait().
                        std::vector<oneapi::tbb::task::suspend_point> arrived;
                        {
                            std::lock_guard<std::mutex> lock{m_mutex};
                            m_suspended.push_back(sp);
                            if(m_suspended.size() == static_cast<std::size_t>(m_threadCount))
                            {
                                arrived.reserve(m_suspended.capacity());
                                arrived.swap(m_suspended);
                            }
                        }

                        // Resume the tasks outside of the critical section.
                        for(auto point : arrived)
                        {
                            oneapi::tbb::task::resume(point);
                        }
                    });
            }

        private:
            TIdx const m_threadCount;
            std::mutex m_mutex;
            std::vector<oneapi::tbb::task::suspend_point> m_suspended;
        };
    } // namespace tbb
} // namespace alpaka::core

#endif
