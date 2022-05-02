/* Copyright 2020 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/Concatenate.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/queue/QueueCpuOmp2Collective.hpp>
#include <alpaka/test/queue/QueueTestFixture.hpp>

#include <catch2/catch.hpp>

#include <atomic>
#include <future>
#include <thread>

using TestQueues = alpaka::meta::Concatenate<
    alpaka::test::TestQueues
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    ,
    std::tuple<std::tuple<alpaka::DevCpu, alpaka::QueueCpuOmp2Collective>>
#endif
    >;

//! Equivalent to CHECK but accounts for potential false negatives
//!
//! This is required when checking if an event or a queue is finished/empty.
//! \warning This macro can be used on host only!
//!
//! \param count number of negative checks to be repeated
//! \param msWait milli seconds to wait if cmd is returning false
//! \param cmd command executed
#define LOOPED_CHECK(count, msWait, cmd)                                                                              \
    do                                                                                                                \
    {                                                                                                                 \
        bool ret = false;                                                                                             \
        for(int i = 0; i < count; ++i)                                                                                \
        {                                                                                                             \
            ret = (cmd);                                                                                              \
            if(ret)                                                                                                   \
                break;                                                                                                \
            std::this_thread::sleep_for(std::chrono::milliseconds(msWait));                                           \
        }                                                                                                             \
        CHECK(ret);                                                                                                   \
    } while(0)

TEMPLATE_LIST_TEST_CASE("queueIsInitiallyEmpty", "[queue]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = alpaka::test::QueueTestFixture<DevQueue>;
    Fixture f;

    CHECK(alpaka::empty(f.m_queue));
}

TEMPLATE_LIST_TEST_CASE("queueCallbackIsWorking", "[queue]", TestQueues)
{
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
    using DevQueue = TestType;
    using Fixture = alpaka::test::QueueTestFixture<DevQueue>;
    Fixture f;

    std::promise<bool> promise;

    alpaka::enqueue(f.m_queue, [&]() { promise.set_value(true); });

    LOOPED_CHECK(30, 100, promise.get_future().get());
#endif
}

TEMPLATE_LIST_TEST_CASE("queueWaitShouldWork", "[queue]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = alpaka::test::QueueTestFixture<DevQueue>;
    Fixture f;

    std::atomic<bool> callbackFinished{false};
    alpaka::enqueue(
        f.m_queue,
        [&callbackFinished]() noexcept
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100u));
            callbackFinished = true;
        });

    alpaka::wait(f.m_queue);
    CHECK(callbackFinished.load() == true);
}

TEMPLATE_LIST_TEST_CASE(
    "queueShouldNotBeEmptyWhenLastTaskIsStillExecutingAndIsEmptyAfterProcessingFinished",
    "[queue]",
    TestQueues)
{
    using DevQueue = TestType;
    using Fixture = alpaka::test::QueueTestFixture<DevQueue>;
    Fixture f;

    std::atomic<bool> callbackFinished{false};
    alpaka::enqueue(
        f.m_queue,
        [&f, &callbackFinished]() noexcept
        {
            LOOPED_CHECK(30, 100, !alpaka::empty(f.m_queue));
            std::this_thread::sleep_for(std::chrono::milliseconds(100u));
            callbackFinished = true;
        });

    // A non-blocking queue will always stay empty because the task has been executed immediately.
    if(!alpaka::test::IsBlockingQueue<typename Fixture::Queue>::value)
    {
        alpaka::wait(f.m_queue);
    }

    CHECK(callbackFinished.load() == true);
    LOOPED_CHECK(30, 100, alpaka::empty(f.m_queue));
}

TEMPLATE_LIST_TEST_CASE("queueShouldNotExecuteTasksInParallel", "[queue]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = alpaka::test::QueueTestFixture<DevQueue>;
    Fixture f;

    std::atomic<bool> taskIsExecuting(false);
    std::promise<void> firstTaskFinished;
    std::future<void> firstTaskFinishedFuture = firstTaskFinished.get_future();
    std::promise<void> secondTaskFinished;
    std::future<void> secondTaskFinishedFuture = secondTaskFinished.get_future();

    std::thread thread1(
        [&f, &taskIsExecuting, &firstTaskFinished]()
        {
            alpaka::enqueue(
                f.m_queue,
                [&taskIsExecuting, &firstTaskFinished]() noexcept
                {
                    CHECK(!taskIsExecuting.exchange(true));
                    std::this_thread::sleep_for(std::chrono::milliseconds(100u));
                    CHECK(taskIsExecuting.exchange(false));
                    firstTaskFinished.set_value();
                });
        });

    std::thread thread2(
        [&f, &taskIsExecuting, &secondTaskFinished]()
        {
            alpaka::enqueue(
                f.m_queue,
                [&taskIsExecuting, &secondTaskFinished]() noexcept
                {
                    CHECK(!taskIsExecuting.exchange(true));
                    std::this_thread::sleep_for(std::chrono::milliseconds(100u));
                    CHECK(taskIsExecuting.exchange(false));
                    secondTaskFinished.set_value();
                });
        });

    // Both tasks have to be enqueued
    thread1.join();
    thread2.join();

    alpaka::wait(f.m_queue);

    firstTaskFinishedFuture.get();
    secondTaskFinishedFuture.get();
}
