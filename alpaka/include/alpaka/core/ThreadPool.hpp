/* Copyright 2023 Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

#include <atomic>
#include <future>
#include <mutex>
#include <optional>
#include <queue>
#include <vector>

namespace alpaka::core::detail
{
    //! A thread pool yielding when there is not enough work to be done.
    struct ThreadPool final
    {
        using Task = std::packaged_task<void()>;

        //! Creates a thread pool with a given thread count
        explicit ThreadPool(std::size_t threadCount)
        {
            if(threadCount < 1)
                throw std::invalid_argument("The argument 'threadCount' has to be greate or equal to one!");
            m_threads.reserve(threadCount);
            for(std::size_t i = 0; i < threadCount; ++i)
                m_threads.emplace_back([this] { threadFunc(); });
        }

        //! Destroys the thread pool, blocking until all enqueued work is done.
        ~ThreadPool()
        {
            m_stop = true; // Signal that concurrent executors should not perform any new work
            for(auto& t : m_threads)
            {
                if(std::this_thread::get_id() == t.get_id())
                {
                    std::cerr << "ERROR in ThreadPool joins itself" << std::endl;
                    std::abort();
                }
                t.join();
            }
        }

        //! Runs the given function on one of the pool in First In First Out (FIFO) order.
        //!
        //! \param task Function object to be called on the pool. Takes an arbitrary number of arguments. Must return
        //!             void.
        //! \param args Arguments for task, cannot be moved. If such parameters must be used, use a lambda and capture
        //!             via move then move the lambda.
        //! \return     A future to the created task.
        template<typename TFnObj, typename... TArgs>
        auto enqueueTask(TFnObj&& task, TArgs&&... args) -> std::future<void>
        {
            auto ptask = Task{[=, t = std::forward<TFnObj>(task)]() noexcept(noexcept(task(args...))) { t(args...); }};
            auto future = ptask.get_future();
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                m_tasks.push(std::move(ptask));
            }
            return future;
        }

    private:
        void threadFunc()
        {
            while(!m_stop.load(std::memory_order_relaxed))
            {
                std::optional<Task> task;
                {
                    std::lock_guard<std::mutex> lock{m_mutex};
                    if(!m_tasks.empty())
                    {
                        task = std::move(m_tasks.front());
                        m_tasks.pop();
                    }
                }
                if(task)
                    (*task)();
                else
                    std::this_thread::yield();
            }
        }

        std::vector<std::thread> m_threads;
        std::queue<Task> m_tasks; // TODO(bgruber): we could consider a lock-free queue here
        std::mutex m_mutex;
        std::atomic<bool> m_stop = false;
    };
} // namespace alpaka::core::detail
