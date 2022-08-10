/* Copyright 2022 Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

namespace alpaka::core
{
    class CallbackThread
    {
        using Task = std::packaged_task<void()>;

    public:
        ~CallbackThread()
        {
            m_stop = true;
            m_cond.notify_one();
            if(m_thread.joinable())
            {
                if(std::this_thread::get_id() == m_thread.get_id())
                {
                    std::cerr << "ERROR in ~CallbackThread: thread joins itself" << std::endl;
                    std::abort();
                }
                m_thread.join();
            }
        }
        auto submit(Task&& newTask) -> std::future<void>
        {
            auto f = newTask.get_future();
            {
                std::unique_lock<std::mutex> lock{m_mutex};
                m_tasks.emplace(std::move(newTask));
                if(!m_thread.joinable())
                    startWorkerThread();
            }
            m_cond.notify_one();
            return f;
        }

    private:
        std::thread m_thread;
        std::condition_variable m_cond;
        std::mutex m_mutex;
        std::atomic<bool> m_stop{false};
        std::queue<Task> m_tasks;

        auto startWorkerThread() -> void
        {
            m_thread = std::thread(
                [this]
                {
                    Task task;
                    while(true)
                    {
                        {
                            std::unique_lock<std::mutex> lock{m_mutex};
                            m_cond.wait(lock, [this] { return m_stop || !m_tasks.empty(); });

                            if(m_stop && m_tasks.empty())
                                break;

                            task = std::move(m_tasks.front());
                            m_tasks.pop();
                        }

                        task();
                    }
                });
        }
    };
} // namespace alpaka::core
