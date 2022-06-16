/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jeffrey Kelling
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>

#include <deque>
#include <functional>
#include <memory>
#include <mutex>

namespace alpaka
{
    namespace detail
    {
        //! The CPU device implementation.
        template<typename TQueue>
        class QueueRegistry
        {
        public:
            ALPAKA_FN_HOST auto getAllExistingQueues() const -> std::vector<std::shared_ptr<TQueue>>
            {
                std::vector<std::shared_ptr<TQueue>> vspQueues;

                std::lock_guard<std::mutex> lk(m_Mutex);
                vspQueues.reserve(std::size(m_queues));

                for(auto it = std::begin(m_queues); it != std::end(m_queues);)
                {
                    auto spQueue = it->lock();
                    if(spQueue)
                    {
                        vspQueues.emplace_back(std::move(spQueue));
                        ++it;
                    }
                    else
                    {
                        it = m_queues.erase(it);
                    }
                }
                return vspQueues;
            }

            //! Registers the given queue on this device.
            //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
            ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<TQueue> spQueue) const -> void
            {
                std::lock_guard<std::mutex> lk(m_Mutex);

                // Register this queue on the device.
                m_queues.push_back(std::move(spQueue));
            }

            using CleanerFunctor = std::function<void()>;
            static ALPAKA_FN_HOST auto registerCleanup(CleanerFunctor cleaner) -> void
            {
                class CleanupList
                {
                    std::mutex m_mutex;
                    std::deque<CleanerFunctor> mutable m_cleanup;

                public:
                    ~CleanupList()
                    {
                        for(auto& c : m_cleanup)
                        {
                            c();
                        }
                    }

                    void push(CleanerFunctor&& c)
                    {
                        std::lock_guard<std::mutex> lk(m_mutex);

                        m_cleanup.emplace_back(std::move(c));
                    }
                };
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors" // running this at exit time is the point
#endif
                static CleanupList cleanupList;
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif

                cleanupList.push(std::move(cleaner));
            }

        private:
            std::mutex mutable m_Mutex;
            std::deque<std::weak_ptr<TQueue>> mutable m_queues;
        };
    } // namespace detail
} // namespace alpaka
