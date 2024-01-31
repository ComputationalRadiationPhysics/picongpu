/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

#include <deque>
#include <functional>
#include <memory>
#include <mutex>

namespace alpaka::detail
{
    //! The CPU/GPU device queue registry implementation.
    //!
    //! @tparam TQueue queue implementation
    template<typename TQueue>
    struct QueueRegistry
    {
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
        ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<TQueue> const& spQueue) const -> void
        {
            std::lock_guard<std::mutex> lk(m_Mutex);

            // Register this queue on the device.
            m_queues.push_back(spQueue);
        }

    private:
        std::mutex mutable m_Mutex;
        std::deque<std::weak_ptr<TQueue>> mutable m_queues;
    };
} // namespace alpaka::detail
