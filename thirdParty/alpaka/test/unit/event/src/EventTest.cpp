/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/event/Traits.hpp>
#include <alpaka/test/event/EventHostManualTrigger.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/queue/QueueCpuOmp2Collective.hpp>
#include <alpaka/test/queue/QueueTestFixture.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using TestQueues = alpaka::meta::Concatenate<
    alpaka::test::TestQueues
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    ,
    std::tuple<std::tuple<alpaka::DevCpu, alpaka::QueueCpuOmp2Collective>>
#endif
    >;

TEMPLATE_LIST_TEST_CASE("eventTestShouldInitiallyBeTrue", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = alpaka::test::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;

    Fixture f;
    alpaka::Event<Queue> event(f.m_dev);

    REQUIRE(alpaka::isComplete(event));
}

TEMPLATE_LIST_TEST_CASE("eventTestShouldBeFalseWhileInQueueAndTrueAfterBeingProcessed", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = alpaka::test::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    Fixture f1;
    if(alpaka::test::isEventHostManualTriggerSupported(f1.m_dev))
    {
        auto q1 = f1.m_queue;
        alpaka::Event<Queue> e1(f1.m_dev);
        alpaka::test::EventHostManualTrigger<Dev> k1(f1.m_dev);

        if(!alpaka::test::IsBlockingQueue<Queue>::value)
        {
            alpaka::enqueue(q1, k1);
        }

        alpaka::enqueue(q1, e1);

        if(!alpaka::test::IsBlockingQueue<Queue>::value)
        {
            REQUIRE(alpaka::isComplete(e1) == false);

            k1.trigger();

            alpaka::wait(q1);
        }

        REQUIRE(alpaka::isComplete(e1));
    }
    else
    {
        std::cerr << "Cannot execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported!"
                  << std::endl;
    }
}

TEMPLATE_LIST_TEST_CASE("eventReEnqueueShouldBePossibleIfNobodyWaitsFor", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = alpaka::test::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!alpaka::test::IsBlockingQueue<Queue>::value)
    {
        Fixture f1;
        if(alpaka::test::isEventHostManualTriggerSupported(f1.m_dev))
        {
            auto q1 = f1.m_queue;
            alpaka::Event<Queue> e1(f1.m_dev);
            alpaka::test::EventHostManualTrigger<Dev> k1(f1.m_dev);
            alpaka::test::EventHostManualTrigger<Dev> k2(f1.m_dev);

            // q1 = [k1]
            alpaka::enqueue(q1, k1);
            REQUIRE(!alpaka::isComplete(k1));

            // q1 = [k1, e1]
            alpaka::enqueue(q1, e1);
            REQUIRE(!alpaka::isComplete(k1));
            REQUIRE(!alpaka::isComplete(e1));

            // q1 = [k1, e1, k2]
            alpaka::enqueue(q1, k2);
            REQUIRE(!alpaka::isComplete(k1));
            REQUIRE(!alpaka::isComplete(e1));
            REQUIRE(!alpaka::isComplete(k2));

            // re-enqueue should be possible
            // q1 = [k1, k2, e1]
            alpaka::enqueue(q1, e1);
            REQUIRE(!alpaka::isComplete(k1));
            REQUIRE(!alpaka::isComplete(k2));
            REQUIRE(!alpaka::isComplete(e1));

            // q1 = [k2, e1]
            k1.trigger();
            REQUIRE(alpaka::isComplete(k1));
            REQUIRE(!alpaka::isComplete(k2));
            REQUIRE(!alpaka::isComplete(e1));

            // q1 = [e1]
            k2.trigger();
            REQUIRE(alpaka::isComplete(k2));
            alpaka::wait(e1);
            REQUIRE(alpaka::isComplete(e1));
        }
        else
        {
            std::cerr << "Cannot execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported by "
                         "the device!"
                      << std::endl;
        }
    }
}

TEMPLATE_LIST_TEST_CASE("eventReEnqueueShouldBePossibleIfSomeoneWaitsFor", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = alpaka::test::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!alpaka::test::IsBlockingQueue<Queue>::value)
    {
        Fixture f1;
        Fixture f2;
        if(alpaka::test::isEventHostManualTriggerSupported(f1.m_dev)
           && alpaka::test::isEventHostManualTriggerSupported(f2.m_dev))
        {
            auto q1 = f1.m_queue;
            auto q2 = f2.m_queue;
            alpaka::Event<Queue> e1(f1.m_dev);
            alpaka::Event<Queue> e2(f2.m_dev);
            alpaka::test::EventHostManualTrigger<Dev> k1(f1.m_dev);
            alpaka::test::EventHostManualTrigger<Dev> k2(f1.m_dev);

            alpaka::enqueue(q1, k1);

            // q1 = [k1]
            REQUIRE(!alpaka::isComplete(k1));

            alpaka::enqueue(q1, e1);

            // q1 = [k1, e1]
            REQUIRE(!alpaka::isComplete(k1));
            REQUIRE(!alpaka::isComplete(e1));

            alpaka::enqueue(q1, k2);

            // q1 = [k1, e1, k2]
            REQUIRE(!alpaka::isComplete(k1));
            REQUIRE(!alpaka::isComplete(e1));
            REQUIRE(!alpaka::isComplete(k2));

            // wait for e1
            alpaka::wait(q2, e1);
            // q2 = [->e1]

            alpaka::enqueue(q2, e2);

            // q2 = [->e1, e2]
            REQUIRE(!alpaka::isComplete(e2));

            // re-enqueue should be possible
            alpaka::enqueue(q1, e1);

            // q1 = [k1, e1, k2, e1_new]
            // q2 = [->e1, e2]
            REQUIRE(!alpaka::isComplete(k1));
            REQUIRE(!alpaka::isComplete(k2));
            REQUIRE(!alpaka::isComplete(e1));
            REQUIRE(!alpaka::isComplete(e2));

            k1.trigger();

            // q1 = [k2, e1_new]
            // q2 = []
            REQUIRE(alpaka::isComplete(k1));
            REQUIRE(!alpaka::isComplete(k2));
            REQUIRE(!alpaka::isComplete(e1));
            REQUIRE(alpaka::isComplete(e2));

            k2.trigger();

            // q1 = []
            // q2 = []
            REQUIRE(alpaka::isComplete(k2));
            alpaka::wait(e1);
            REQUIRE(alpaka::isComplete(e1));
            alpaka::wait(e2);
            REQUIRE(alpaka::isComplete(e2));
        }
        else
        {
            std::cerr << "Cannot execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported by "
                         "the device!"
                      << std::endl;
        }
    }
}

// github issue #388
TEMPLATE_LIST_TEST_CASE("waitForEventThatAlreadyFinishedShouldBeSkipped", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = alpaka::test::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!alpaka::test::IsBlockingQueue<Queue>::value)
    {
        Fixture f1;
        Fixture f2;
        if(alpaka::test::isEventHostManualTriggerSupported(f1.m_dev)
           && alpaka::test::isEventHostManualTriggerSupported(f2.m_dev))
        {
            auto q1 = f1.m_queue;
            auto q2 = f2.m_queue;
            alpaka::test::EventHostManualTrigger<Dev> k1(f1.m_dev);
            alpaka::test::EventHostManualTrigger<Dev> k2(f2.m_dev);
            alpaka::Event<Queue> e1(f1.m_dev);

            // 1. kernel k1 is enqueued into queue q1
            alpaka::enqueue(q1, k1);
            // q1 = [k1]

            // 2. kernel k2 is enqueued into queue q2
            alpaka::enqueue(q2, k2);
            // q2 = [k2]

            // 3. event e1 is enqueued into queue q1
            alpaka::enqueue(q1, e1);
            // q1 = [k1, e1]

            // 4. q2 waits for e1
            alpaka::wait(q2, e1);
            // q2 = [k2, ->e1]

            // 5. kernel k1 finishes
            k1.trigger();
            // q1 = [e1]

            // 6. e1 is finished
            alpaka::wait(e1);

            // q1 = []
            // q2 = [k2, ->e1]
            REQUIRE(!alpaka::isComplete(k2));
            REQUIRE(alpaka::isComplete(e1));

            // 7. e1 is re-enqueued again but this time into q2
            alpaka::enqueue(q2, e1);

            // q2 = [k2, ->e1, e1]
            REQUIRE(!alpaka::isComplete(k2));
            REQUIRE(!alpaka::isComplete(e1));

            // 8. kernel k2 finishes
            k2.trigger();
            // q2 = []
            REQUIRE(alpaka::isComplete(e1));

            // 9. e1 had already been signaled so there should not be waited even though the event is now reused within
            // q2 and its current state is 'unfinished' again. q2 = [e1]

            // Both queues should successfully finish
            alpaka::wait(q1);
            alpaka::wait(q2);
        }
        else
        {
            std::cerr << "Cannot execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported by "
                         "the device!"
                      << std::endl;
        }
    }
}

TEMPLATE_LIST_TEST_CASE("eventReEnqueueWithSomeoneWaitsForEventInOrderLifetimeRelease", "[event]", TestQueues)
{
    // A re-enqueued event will be released in the order it is recorded. The tests validate that dependencies between
    // queues and the re-enqueued event will be correct.
    using DevQueue = TestType;
    using Fixture = alpaka::test::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!alpaka::test::IsBlockingQueue<Queue>::value)
    {
        Fixture f1;
        Fixture f2;
        Fixture f3;
        if(alpaka::test::isEventHostManualTriggerSupported(f1.m_dev)
           && alpaka::test::isEventHostManualTriggerSupported(f2.m_dev)
           && alpaka::test::isEventHostManualTriggerSupported(f3.m_dev))
        {
            auto q1 = f1.m_queue;
            auto q2 = f2.m_queue;
            auto q3 = f3.m_queue;

            alpaka::Event<Queue> e1(f1.m_dev);
            alpaka::Event<Queue> e2(f2.m_dev);
            alpaka::Event<Queue> e3(f3.m_dev);
            alpaka::test::EventHostManualTrigger<Dev> k1_0(f1.m_dev);
            alpaka::test::EventHostManualTrigger<Dev> k1_1(f1.m_dev);
            alpaka::test::EventHostManualTrigger<Dev> k2(f2.m_dev);
            alpaka::test::EventHostManualTrigger<Dev> k3(f3.m_dev);

            alpaka::enqueue(q1, k1_0);
            alpaka::enqueue(q1, e1);
            // q1 = [k1_0,e1]
            alpaka::enqueue(q2, k2);
            // q2 = [k2]
            REQUIRE(!alpaka::isComplete(k1_0));

            alpaka::wait(q2, e1);
            alpaka::enqueue(q2, e2);
            // q2 = [k2,->e1,e2]
            alpaka::enqueue(q1, k1_1);
            alpaka::enqueue(q1, e1);
            // q1 = [k1_0,e1,k1_1,e1_new]
            alpaka::enqueue(q3, k3);
            alpaka::wait(q3, e1);
            alpaka::enqueue(q3, e3);

            // q1 = [k1_0,e1,k1_1,e1_new]
            // q2 = [k2,->e1,e2]
            // q3 = [k3,->e1_new,e3]
            REQUIRE(!alpaka::isComplete(k1_0));
            REQUIRE(!alpaka::isComplete(k1_1));
            REQUIRE(!alpaka::isComplete(k2));
            REQUIRE(!alpaka::isComplete(k3));
            REQUIRE(!alpaka::isComplete(e1));
            REQUIRE(!alpaka::isComplete(e2));
            REQUIRE(!alpaka::isComplete(e3));

            k3.trigger();

            // q1 = [k1_0,e1,k1_1,e1_new]
            // q2 = [k2,->e1,e2]
            // q3 = [->e1_new,e3]
            REQUIRE(!alpaka::isComplete(k1_0));
            REQUIRE(!alpaka::isComplete(k1_1));
            REQUIRE(!alpaka::isComplete(k2));
            REQUIRE(alpaka::isComplete(k3));
            REQUIRE(!alpaka::isComplete(e1));
            REQUIRE(!alpaka::isComplete(e2));
            REQUIRE(!alpaka::isComplete(e3));

            k2.trigger();

            // q1 = [k1_0,e1,k1_1,e1_new]
            // q2 = [->e1,e2]
            // q3 = [->e1_new,e3]
            REQUIRE(!alpaka::isComplete(k1_0));
            REQUIRE(!alpaka::isComplete(k1_1));
            REQUIRE(alpaka::isComplete(k2));
            REQUIRE(alpaka::isComplete(k3));
            REQUIRE(!alpaka::isComplete(e1));
            REQUIRE(!alpaka::isComplete(e2));
            REQUIRE(!alpaka::isComplete(e3));

            // After the kernel k1_0 is released e3 is not allowed to be ready because q3 depends on the oldest e1_new
            // state.
            k1_0.trigger();
            // q1 = [k1_1,e1_new]
            // q2 = []
            // q3 = [->e1_new,e3]

            REQUIRE(alpaka::isComplete(k1_0));
            REQUIRE(!alpaka::isComplete(k1_1));
            REQUIRE(alpaka::isComplete(k2));
            REQUIRE(alpaka::isComplete(k3));
            REQUIRE(!alpaka::isComplete(e1));
            REQUIRE(alpaka::isComplete(e2));
            REQUIRE(!alpaka::isComplete(e3));

            k1_1.trigger();
            // q1 = []
            // q2 = []
            // q3 = []
            REQUIRE(alpaka::isComplete(k1_0));
            REQUIRE(alpaka::isComplete(k1_1));
            REQUIRE(alpaka::isComplete(k2));
            REQUIRE(alpaka::isComplete(k3));
            REQUIRE(alpaka::isComplete(e1));
            REQUIRE(alpaka::isComplete(e2));
            REQUIRE(alpaka::isComplete(e3));
        }
        else
        {
            std::cerr << "Cannot execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported by "
                         "the device!"
                      << std::endl;
        }
    }
}

TEMPLATE_LIST_TEST_CASE("eventReEnqueueWithSomeoneWaitsForEventOutOfOrderLifetimeRelease", "[event]", TestQueues)
{
    // A re-enqueued event will be released in the opposite order it is recorded.
    // The tests validate that dependencies between queues and the re-enqueued event will be correct.
    using DevQueue = TestType;
    using Fixture = alpaka::test::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!alpaka::test::IsBlockingQueue<Queue>::value)
    {
        Fixture f1;
        Fixture f2;
        Fixture f3;
        if(alpaka::test::isEventHostManualTriggerSupported(f1.m_dev)
           && alpaka::test::isEventHostManualTriggerSupported(f2.m_dev))
        {
            auto q1 = f1.m_queue;
            auto q2 = f2.m_queue;
            auto q3 = f3.m_queue;
            alpaka::Event<Queue> e1(f1.m_dev);
            alpaka::Event<Queue> e2(f2.m_dev);
            alpaka::Event<Queue> e3(f3.m_dev);
            alpaka::test::EventHostManualTrigger<Dev> k1_0(f1.m_dev);
            alpaka::test::EventHostManualTrigger<Dev> k2(f2.m_dev);

            alpaka::enqueue(q1, k1_0);
            alpaka::enqueue(q1, e1);
            // q1 = [k1_0,e1]
            alpaka::enqueue(q2, k2);
            // q2 = [k2]
            REQUIRE(!alpaka::isComplete(k1_0));

            alpaka::wait(q3, e1);
            alpaka::enqueue(q3, e3);
            // q3 = [->e1,e3]

            alpaka::enqueue(q2, e1);
            // q2 = [k2,e1_new]

            alpaka::enqueue(q2, e2);
            // q1 = [k1_0,e1]
            // q2 = [k2,e1_new,e2]
            // q3 = [->e1,e3]

            REQUIRE(!alpaka::isComplete(k1_0));
            REQUIRE(!alpaka::isComplete(k2));
            REQUIRE(!alpaka::isComplete(e1));
            REQUIRE(!alpaka::isComplete(e2));
            REQUIRE(!alpaka::isComplete(e3));

            // We release first the kernel which is blocking the most recent enqueue of event e1.
            // q3 is not allowed to be freed because this queue depends on the oldest enqueue of e1.
            k2.trigger();

            // q1 = [k1_0,e1]
            // q2 = []
            // q3 = [->e1,e3]

            REQUIRE(!alpaka::isComplete(k1_0));

            REQUIRE(alpaka::isComplete(k2));
            REQUIRE(alpaka::isComplete(e1));
            REQUIRE(alpaka::isComplete(e2));
            REQUIRE(!alpaka::isComplete(e3));


            k1_0.trigger();

            // q1 = []
            // q2 = []
            // q3 = []
            REQUIRE(alpaka::isComplete(k1_0));

            REQUIRE(alpaka::isComplete(k2));
            REQUIRE(alpaka::isComplete(e1));
            REQUIRE(alpaka::isComplete(e2));
            REQUIRE(alpaka::isComplete(e3));

            alpaka::wait(q1);
            alpaka::wait(q2);
            alpaka::wait(q3);
        }
        else
        {
            std::cerr << "Cannot execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported by "
                         "the device!"
                      << std::endl;
        }
    }
}
