/* Copyright 2022 Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

// Clang does not support exceptions when natively compiling device code.
// This is no problem at some places but others explicitly rely on std::exception_ptr,
// std::current_exception, std::make_exception_ptr, etc. which are not declared in device code.
// Therefore, we can not even parse those parts when compiling device code.
#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/ThreadTraits.hpp>

#include <atomic>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace alpaka::core
{
    namespace detail
    {
        template<typename T>
        struct ThreadSafeQueue
        {
            ThreadSafeQueue() = default;

            [[nodiscard]] auto empty() const -> bool
            {
                return m_queue.empty();
            }

            //! Pushes the given value onto the back of the queue.
            void push(T&& t)
            {
                std::lock_guard<std::mutex> lk(m_mutex);
                m_queue.push(std::move(t));
            }

            //! Pops the given value from the front of the queue.
            auto pop(T& t) -> bool
            {
                std::lock_guard<std::mutex> lk(m_mutex);

                if(m_queue.empty())
                    return false;
                t = std::move(m_queue.front());
                m_queue.pop();
                return true;
            }

        private:
            std::queue<T> m_queue;
            std::mutex m_mutex;
        };

        //! ITaskPkg.
        // \NOTE: We can not use std::packaged_task as it forces the use of std::future.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wweak-vtables"
#endif
        struct ITaskPkg
        {
            virtual ~ITaskPkg() = default;

            //! Runs this task.
            void runTask() noexcept
            {
                try
                {
                    run();
                }
                catch(...)
                {
// Workaround: Clang can not support this when natively compiling device code.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                    setException(std::current_exception());
#endif
                }
            }

// Workaround: Clang can not support this when natively compiling device code.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
            //! Sets an exception.
            virtual auto setException(std::exception_ptr const& exceptPtr) -> void = 0;
#endif

        protected:
            //! The execution function.
            virtual auto run() -> void = 0;
        };
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif

        //! \tparam TPromise The promise type returned by the task.
        //! \tparam TFnObj The type of the function to execute.
        template<template<typename> class TPromise, typename TFnObj>
        struct TaskPkg final : ITaskPkg
        {
            using TFnObjReturn = decltype(std::declval<TFnObj>()());

            TaskPkg(TFnObj&& func) : m_Promise(), m_FnObj(std::move(func))
            {
            }

// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
            //! Sets an exception.
            void setException(std::exception_ptr const& exceptPtr) final
            {
                m_Promise.set_exception(exceptPtr);
            }
#endif
            TPromise<TFnObjReturn> m_Promise;

        private:
            //! The execution function.
            void run() final
            {
                if constexpr(std::is_void_v<TFnObjReturn>)
                {
                    this->m_FnObj();
                    m_Promise.set_value();
                }
                else
                    m_Promise.set_value(this->m_FnObj());
            }

            // NOTE: To avoid invalid memory accesses to memory of a different thread
            // `std::remove_reference` enforces the function object to be copied.
            std::remove_reference_t<TFnObj> m_FnObj;
        };

        template<typename TFnObj0, typename TFnObj1>
        auto invokeBothReturnFirst(TFnObj0&& fn0, TFnObj1&& fn1)
        {
            if constexpr(!std::is_same_v<void, decltype(std::declval<TFnObj0>()())>)
            {
                auto ret = fn0();
                fn1();
                return ret;
            }
            else
            {
                fn0();
                fn1();
            }
        }

        template<typename TMutex, typename TCondVar>
        struct ConcurrentExecPoolMutexAndCond
        {
            TMutex m_mtxWakeup;
            TCondVar m_cvWakeup;
        };

        struct Empty
        {
        };

        //! ConcurrentExecPool using yield or a condition variable to wait for new work.
        //!
        //! \tparam TConcurrentExec The type of concurrent executor (for example std::thread).
        //! \tparam TPromise The promise type returned by the task.
        //! \tparam TYield The type is required to have a static method "void yield()" to yield the current thread
        //! if there is no work.
        //! \tparam TMutex The mutex type used for locking threads.
        //! \tparam TCondVar The condition variable type used to make the threads wait if there is no work.
        //! \tparam TisYielding Boolean value if the threads should yield instead of wait for a condition variable.
        template<
            typename TIdx,
            typename TConcurrentExec,
            template<typename TFnObjReturn>
            typename TPromise,
            typename TYield,
            typename TMutex = void,
            typename TCondVar = void,
            bool TisYielding = true>
        struct ConcurrentExecPool final
            : std::conditional_t<TisYielding, Empty, ConcurrentExecPoolMutexAndCond<TMutex, TCondVar>>
        {
            //! Creates a concurrent executors pool with a specific number of concurrent executors and a maximum
            //! number of queued tasks.
            //!
            //! \param concurrentExecutionCount
            //!    The guaranteed number of concurrent executors used in the pool.
            //!    This is also the maximum number of tasks worked on concurrently.
            ConcurrentExecPool(TIdx concurrentExecutionCount)
            {
                if(concurrentExecutionCount < 1)
                {
                    throw std::invalid_argument(
                        "The argument 'concurrentExecutionCount' has to be greate or equal to one!");
                }

                m_vConcurrentExecs.reserve(static_cast<std::size_t>(concurrentExecutionCount));

                // Create all concurrent executors.
                for(TIdx i = 0; i < concurrentExecutionCount; ++i)
                    m_vConcurrentExecs.emplace_back([this]() { concurrentExecFn(); });
            }

            ConcurrentExecPool(ConcurrentExecPool const&) = delete;
            auto operator=(ConcurrentExecPool const&) -> ConcurrentExecPool& = delete;

            //! Completes any currently running task normally.
            //! Signals a std::runtime_error exception to any other tasks that was not able to run.
            ~ConcurrentExecPool()
            {
                // Signal that concurrent executors should not perform any new work
                if constexpr(TisYielding)
                    m_bShutdownFlag.store(true);
                else
                {
                    {
                        std::unique_lock<TMutex> lock(this->m_mtxWakeup);
                        m_bShutdownFlag = true;
                    }
                    this->m_cvWakeup.notify_all();
                }

                joinAllConcurrentExecs();

                // Signal to each incomplete task that it will not complete due to pool destruction.
                while(auto task = popTask())
                {
                    auto const except
                        = std::runtime_error("Could not perform task before ConcurrentExecPool destruction");
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                    task->setException(std::make_exception_ptr(except));
#endif
                }
            }

            //! Runs the given function on one of the pool in First In First Out (FIFO) order.
            //!
            //! \tparam TFnObj  The function type.
            //! \param task     Function object to be called on the pool.
            //!                 Takes an arbitrary number of arguments and arbitrary return type.
            //! \tparam TArgs   The argument types pack.
            //! \param args     Arguments for task, cannot be moved.
            //!                 If such parameters must be used, use a lambda and capture via move then move the
            //!                 lambda.
            //!
            //! \return Signals when the task has completed with either success or an exception.
            //!         Also results in an exception if the pool is destroyed before execution has begun.
            template<typename TFnObj, typename... TArgs>
            auto enqueueTask(TFnObj&& task, TArgs&&... args)
            {
                auto boundTask = [=]() { return task(args...); };
                auto decrementNumActiveTasks = [this]() { --m_numActiveTasks; };

                auto extendedTask = [boundTask, decrementNumActiveTasks]()
                { return invokeBothReturnFirst(std::move(boundTask), std::move(decrementNumActiveTasks)); };

                using TaskPackage = TaskPkg<TPromise, decltype(extendedTask)>;
                auto pTaskPackage = new TaskPackage(std::move(extendedTask));
                std::shared_ptr<ITaskPkg> upTaskPackage(pTaskPackage);

                auto future = pTaskPackage->m_Promise.get_future();

                ++m_numActiveTasks;
                if constexpr(TisYielding)
                    m_qTasks.push(std::move(upTaskPackage));
                else
                {
                    {
                        std::lock_guard<TMutex> lock(this->m_mtxWakeup);
                        m_qTasks.push(std::move(upTaskPackage));
                    }

                    this->m_cvWakeup.notify_one();
                }

                return future;
            }

            //! \return The number of concurrent executors available.
            [[nodiscard]] auto getConcurrentExecutionCount() const -> TIdx
            {
                return std::size(m_vConcurrentExecs);
            }

            //! \return If the thread pool is idle.
            [[nodiscard]] auto isIdle() const -> bool
            {
                return m_numActiveTasks == 0u;
            }

            void detach(std::shared_ptr<ConcurrentExecPool>&& self)
            {
                m_self = std::move(self);
                if constexpr(TisYielding)
                    m_bDetachedFlag = true;
                else
                {
                    std::lock_guard<TMutex> lock(this->m_mtxWakeup);
                    m_bDetachedFlag = true;
                    // we need to notify during the lock, because setting m_bDetachedFlag to true, allows another
                    // thread to delete this and thus destroy m_cvWakeup.
                    this->m_cvWakeup.notify_one();
                }
            }

            auto takeDetachHandle() -> std::shared_ptr<ConcurrentExecPool>
            {
                if(m_bDetachedFlag.exchange(false))
                    return std::move(m_self);
                else
                    return nullptr;
            }

        private:
            //! The function the concurrent executors are executing.
            void concurrentExecFn()
            {
                // Checks whether pool is being destroyed, if so, stop running.
                while(!m_bShutdownFlag.load(std::memory_order_relaxed))
                {
                    if constexpr(TisYielding)
                    {
                        if(auto task = popTask())
                            task->runTask();
                        else
                        {
                            if(takeDetachHandle())
                                return; // Pool was detached and is idle, stop and delete
                            TYield::yield();
                        }
                    }
                    else
                    {
                        if(auto task = popTask())
                            task->runTask();

                        std::unique_lock<TMutex> lock(this->m_mtxWakeup);
                        if(m_qTasks.empty())
                        {
                            auto self = takeDetachHandle();
                            if(self)
                            {
                                // Pool was detached and is idle, stop and delete
                                lock.unlock(); // TODO(bgruber): I guess we unlock here so the mutex is not locked when
                                               // the dtor of self runs, which also tries to lock?
                                return;
                            }

                            // If the shutdown flag has been set since the last check, return now.
                            if(m_bShutdownFlag)
                                return;

                            this->m_cvWakeup.wait(
                                lock,
                                [this] { return !m_qTasks.empty() || m_bShutdownFlag || m_bDetachedFlag; });
                        }
                    }
                }
            }

            //! Joins all concurrent executors.
            void joinAllConcurrentExecs()
            {
                for(auto&& concurrentExec : m_vConcurrentExecs)
                {
                    if(isThisThread(concurrentExec))
                        concurrentExec.detach();
                    else
                        concurrentExec.join();
                }
            }

            //! Pops a task from the queue.
            auto popTask() -> std::shared_ptr<ITaskPkg>
            {
                std::shared_ptr<ITaskPkg> out;
                if(m_qTasks.pop(out))
                    return out;
                else
                    return nullptr;
            }

        private:
            std::vector<TConcurrentExec> m_vConcurrentExecs;
            ThreadSafeQueue<std::shared_ptr<ITaskPkg>> m_qTasks;
            std::atomic<std::uint32_t> m_numActiveTasks = 0u;
            std::atomic<bool> m_bShutdownFlag = false;
            std::atomic<bool> m_bDetachedFlag = false;
            std::shared_ptr<ConcurrentExecPool> m_self = nullptr;
        };
    } // namespace detail
} // namespace alpaka::core
