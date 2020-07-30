/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/event/Traits.hpp>

#include <alpaka/test/event/EventHostManualTrigger.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/queue/QueueTestFixture.hpp>
#include <alpaka/test/queue/QueueCpuOmp2Collective.hpp>

#include <catch2/catch.hpp>

using TestQueues = alpaka::meta::Concatenate<
        alpaka::test::queue::TestQueues
 #ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
        ,
        std::tuple<std::tuple<alpaka::dev::DevCpu, alpaka::queue::QueueCpuOmp2Collective>>
#endif
    >;

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "eventTestShouldInitiallyBeTrue", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = alpaka::test::queue::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;

    Fixture f;
    alpaka::event::Event<Queue> event(f.m_dev);

    REQUIRE(alpaka::event::test(event));
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "eventTestShouldBeFalseWhileInQueueAndTrueAfterBeingProcessed", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = alpaka::test::queue::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    Fixture f1;
    if(alpaka::test::event::isEventHostManualTriggerSupported(f1.m_dev))
    {
        auto q1 = f1.m_queue;
        alpaka::event::Event<Queue> e1(f1.m_dev);
        alpaka::test::event::EventHostManualTrigger<Dev> k1(f1.m_dev);

        if(!alpaka::test::queue::IsBlockingQueue<Queue>::value)
        {
            alpaka::queue::enqueue(q1, k1);
        }

        alpaka::queue::enqueue(q1, e1);

        if(!alpaka::test::queue::IsBlockingQueue<Queue>::value)
        {
            REQUIRE(alpaka::event::test(e1) == false);

            k1.trigger();

            alpaka::wait::wait(q1);
        }

        REQUIRE(alpaka::event::test(e1));
    }
    else
    {
        std::cerr << "Can not execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported!" << std::endl;
    }
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "eventReEnqueueShouldBePossibleIfNobodyWaitsFor", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = alpaka::test::queue::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!alpaka::test::queue::IsBlockingQueue<Queue>::value)
    {
        Fixture f1;
        if(alpaka::test::event::isEventHostManualTriggerSupported(f1.m_dev))
        {
            auto q1 = f1.m_queue;
            alpaka::event::Event<Queue> e1(f1.m_dev);
            alpaka::test::event::EventHostManualTrigger<Dev> k1(f1.m_dev);
            alpaka::test::event::EventHostManualTrigger<Dev> k2(f1.m_dev);

            // q1 = [k1]
            alpaka::queue::enqueue(q1, k1);
            REQUIRE(!alpaka::event::test(k1));

            // q1 = [k1, e1]
            alpaka::queue::enqueue(q1, e1);
            REQUIRE(!alpaka::event::test(k1));
            REQUIRE(!alpaka::event::test(e1));

            // q1 = [k1, e1, k2]
            alpaka::queue::enqueue(q1, k2);
            REQUIRE(!alpaka::event::test(k1));
            REQUIRE(!alpaka::event::test(e1));
            REQUIRE(!alpaka::event::test(k2));

            // re-enqueue should be possible
            // q1 = [k1, k2, e1]
            alpaka::queue::enqueue(q1, e1);
            REQUIRE(!alpaka::event::test(k1));
            REQUIRE(!alpaka::event::test(k2));
            REQUIRE(!alpaka::event::test(e1));

            // q1 = [k2, e1]
            k1.trigger();
            REQUIRE(alpaka::event::test(k1));
            REQUIRE(!alpaka::event::test(k2));
            REQUIRE(!alpaka::event::test(e1));

            // q1 = [e1]
            k2.trigger();
            REQUIRE(alpaka::event::test(k2));
            alpaka::wait::wait(e1);
            REQUIRE(alpaka::event::test(e1));
        }
        else
        {
            std::cerr << "Can not execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported!" << std::endl;
        }
    }
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "eventReEnqueueShouldBePossibleIfSomeoneWaitsFor", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = alpaka::test::queue::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!alpaka::test::queue::IsBlockingQueue<Queue>::value)
    {
        Fixture f1;
        Fixture f2;
        if(alpaka::test::event::isEventHostManualTriggerSupported(f1.m_dev)
            && alpaka::test::event::isEventHostManualTriggerSupported(f2.m_dev))
        {
            auto q1 = f1.m_queue;
            auto q2 = f2.m_queue;
            alpaka::event::Event<Queue> e1(f1.m_dev);
            alpaka::event::Event<Queue> e2(f2.m_dev);
            alpaka::test::event::EventHostManualTrigger<Dev> k1(f1.m_dev);
            alpaka::test::event::EventHostManualTrigger<Dev> k2(f1.m_dev);

            // q1 = [k1]
            alpaka::queue::enqueue(q1, k1);
            REQUIRE(!alpaka::event::test(k1));

            // q1 = [k1, e1]
            alpaka::queue::enqueue(q1, e1);
            REQUIRE(!alpaka::event::test(k1));
            REQUIRE(!alpaka::event::test(e1));

            // q1 = [k1, e1, k2]
            alpaka::queue::enqueue(q1, k2);
            REQUIRE(!alpaka::event::test(k1));
            REQUIRE(!alpaka::event::test(e1));
            REQUIRE(!alpaka::event::test(k2));

            // wait for e1
            // q2 = [->e1]
            alpaka::wait::wait(q2, e1);

            // q2 = [->e1, e2]
            alpaka::queue::enqueue(q2, e2);
            REQUIRE(!alpaka::event::test(e2));

            // re-enqueue should be possible
            // q1 = [k1, e1-old, k2, e1]
            alpaka::queue::enqueue(q1, e1);
            REQUIRE(!alpaka::event::test(k1));
            REQUIRE(!alpaka::event::test(k2));
            REQUIRE(!alpaka::event::test(e1));
            REQUIRE(!alpaka::event::test(e2));

            // q1 = [k2, e1]
            k1.trigger();
            REQUIRE(alpaka::event::test(k1));
            REQUIRE(!alpaka::event::test(k2));
            REQUIRE(!alpaka::event::test(e1));
            REQUIRE(!alpaka::event::test(e2));

            // q1 = [e1]
            k2.trigger();
            REQUIRE(alpaka::event::test(k2));
            alpaka::wait::wait(e1);
            REQUIRE(alpaka::event::test(e1));
            alpaka::wait::wait(e2);
            REQUIRE(alpaka::event::test(e2));
        }
        else
        {
            std::cerr << "Can not execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported!" << std::endl;
        }
    }
}


//-----------------------------------------------------------------------------
// github issue #388
TEMPLATE_LIST_TEST_CASE( "waitForEventThatAlreadyFinishedShouldBeSkipped", "[event]", TestQueues)
{
    using DevQueue = TestType;
    using Fixture = alpaka::test::queue::QueueTestFixture<DevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!alpaka::test::queue::IsBlockingQueue<Queue>::value)
    {
        Fixture f1;
        Fixture f2;
        if(alpaka::test::event::isEventHostManualTriggerSupported(f1.m_dev)
            && alpaka::test::event::isEventHostManualTriggerSupported(f2.m_dev))
        {
            auto q1 = f1.m_queue;
            auto q2 = f2.m_queue;
            alpaka::test::event::EventHostManualTrigger<Dev> k1(f1.m_dev);
            alpaka::test::event::EventHostManualTrigger<Dev> k2(f2.m_dev);
            alpaka::event::Event<Queue> e1(f1.m_dev);

            // 1. kernel k1 is enqueued into queue q1
            // q1 = [k1]
            alpaka::queue::enqueue(q1, k1);
            // 2. kernel k2 is enqueued into queue q2
            // q2 = [k2]
            alpaka::queue::enqueue(q2, k2);

            // 3. event e1 is enqueued into queue q1
            // q1 = [k1, e1]
            alpaka::queue::enqueue(q1, e1);

            // 4. q2 waits for e1
            // q2 = [k2, ->e1]
            alpaka::wait::wait(q2, e1);

            // 5. kernel k1 finishes
            // q1 = [e1]
            k1.trigger();

            // 6. e1 is finished
            // q1 = []
            alpaka::wait::wait(e1);
            REQUIRE(alpaka::event::test(e1));

            // 7. e1 is re-enqueued again but this time into q2
            // q2 = [k2, ->e1, e1]
            alpaka::queue::enqueue(q2, e1);

            // 8. kernel k2 finishes
            // q2 = [->e1, e1]
            k2.trigger();

            // 9. e1 had already been signaled so there should not be waited even though the event is now reused within q2 and its current state is 'unfinished' again.
            // q2 = [e1]

            // Both queues should successfully finish
            alpaka::wait::wait(q1);
            // q2 = []
            alpaka::wait::wait(q2);
        }
        else
        {
            std::cerr << "Can not execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported!" << std::endl;
        }
    }
}
