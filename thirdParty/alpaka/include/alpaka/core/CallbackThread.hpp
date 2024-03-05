/* Copyright 2022 Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"

#include <cassert>
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
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wweak-vtables"
#endif
        // A custom class is used because std::function<F> requires F to be copyable, and std::packaged_task provides a
        // std::future which will keep the task alive and we cannot control the moment the future is set.
        //! \todo with C++23 std::move_only_function should be used
        struct Task
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
        {
            virtual ~Task() = default;
            virtual void run() = 0;
        };

        template<typename Function>
        struct FunctionHolder : Task
        {
            Function m_func;

            template<typename FunctionFwd>
            explicit FunctionHolder(FunctionFwd&& func) : m_func{std::forward<FunctionFwd>(func)}
            {
            }

            void run() override
            {
                // if m_func throws, let it propagate
                m_func();
            }
        };

        using TaskPackage = std::pair<std::unique_ptr<Task>, std::promise<void>>;

    public:
        ~CallbackThread()
        {
            {
                std::unique_lock<std::mutex> lock{m_mutex};
                m_stop = true;
                m_cond.notify_one();
            }

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

        //! It is guaranteed that the task is fully destroyed before the future's result is set.
        //! @{
        template<typename NullaryFunction>
        auto submit(NullaryFunction&& nf) -> std::future<void>
        {
            using DecayedFunction = std::decay_t<NullaryFunction>;
            static_assert(
                std::is_void_v<std::invoke_result_t<DecayedFunction>>,
                "Submitted function must not have any arguments and return void.");

            // FunctionHolder stores a copy of the user's task, but may be constructed from an expiring value to avoid
            // the copy. We do NOT store a reference to the users task, which could dangle if the user isn't careful.
            auto tp = std::pair(
                std::unique_ptr<Task>(new FunctionHolder<DecayedFunction>{std::forward<NullaryFunction>(nf)}),
                std::promise<void>{});
            auto f = tp.second.get_future();
            {
                std::unique_lock<std::mutex> lock{m_mutex};
                m_tasks.emplace(std::move(tp));
                if(!m_thread.joinable())
                    startWorkerThread();
                m_cond.notify_one();
            }

            return f;
        }

        //! @}

        //! @return True if queue is empty and no task is executed else false.
        //! If only one tasks is enqueued and the task is executed the task will see the queue as not empty.
        //! During the destruction of this single enqueued task the queue will already be accounted as empty.
        [[nodiscard]] auto empty()
        {
            std::unique_lock<std::mutex> lock{m_mutex};
            return m_tasks.empty();
        }

    private:
        std::thread m_thread;
        std::condition_variable m_cond;
        std::mutex m_mutex;
        bool m_stop{false};
        std::queue<TaskPackage> m_tasks;

        auto startWorkerThread() -> void
        {
            m_thread = std::thread(
                [this]
                {
                    while(true)
                    {
                        std::promise<void> taskPromise;
                        std::exception_ptr eptr;
                        {
                            // Task is destroyed before promise is updated but after the queue state is up to date.
                            std::unique_ptr<Task> task = nullptr;
                            {
                                std::unique_lock<std::mutex> lock{m_mutex};
                                m_cond.wait(lock, [this] { return m_stop || !m_tasks.empty(); });

                                if(m_stop && m_tasks.empty())
                                    break;

                                task = std::move(m_tasks.front().first);
                                taskPromise = std::move(m_tasks.front().second);
                            }
                            assert(task);
                            try
                            {
                                task->run();
                            }
                            catch(...)
                            {
                                eptr = std::current_exception();
                            }
                            {
                                std::unique_lock<std::mutex> lock{m_mutex};
                                // Pop empty data from the queue, task and promise will be destroyed later in a
                                // well-defined order.
                                m_tasks.pop();
                            }
                            // Task will be destroyed here, the queue status is already updated.
                        }
                        // In case the executed tasks is the last task in the queue the waiting threads will see the
                        // queue as empty.
                        if(eptr)
                            taskPromise.set_exception(std::move(eptr));
                        else
                            taskPromise.set_value();
                    }
                });
        }
    };
} // namespace alpaka::core
