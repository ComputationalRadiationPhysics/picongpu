/**
 * \file
 * Copyright 2017 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

// \Hack: Boost.MPL defines BOOST_MPL_CFG_GPU_ENABLED to __host__ __device__ if nvcc is used.
// BOOST_AUTO_TEST_CASE_TEMPLATE and its internals are not GPU enabled but is using boost::mpl::for_each internally.
// For each template parameter this leads to:
// /home/travis/build/boost/boost/mpl/for_each.hpp(78): warning: calling a __host__ function from a __host__ __device__ function is not allowed
// because boost::mpl::for_each has the BOOST_MPL_CFG_GPU_ENABLED attribute but the test internals are pure host methods.
// Because we do not use MPL within GPU code here, we can disable the MPL GPU support.
#define BOOST_MPL_CFG_GPU_ENABLED

#include <alpaka/alpaka.hpp>
#include <alpaka/test/event/EventHostManualTrigger.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/queue/QueueTestFixture.hpp>

#include <alpaka/core/BoostPredef.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

BOOST_AUTO_TEST_SUITE(event)

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    eventTestShouldInitiallyBeTrue,
    TDevQueue,
    alpaka::test::queue::TestQueues)
{
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    using Queue = typename Fixture::Queue;

    Fixture f;
    alpaka::event::Event<Queue> event(f.m_dev);

    BOOST_REQUIRE_EQUAL(
        true,
        alpaka::event::test(event));
}

using TestQueues = alpaka::test::queue::TestQueues;

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    eventTestShouldBeFalseWhileInQueueAndTrueAfterBeingProcessed,
    TDevQueue,
    TestQueues)
{
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    Fixture f1;
    if(alpaka::test::event::isEventHostManualTriggerSupported(f1.m_dev))
    {
        auto q1 = f1.m_queue;
        alpaka::event::Event<Queue> e1(f1.m_dev);
        alpaka::test::event::EventHostManualTrigger<Dev> k1(f1.m_dev);

        if(!alpaka::test::queue::IsSyncQueue<Queue>::value)
        {
            alpaka::queue::enqueue(q1, k1);
        }

        alpaka::queue::enqueue(q1, e1);

        if(!alpaka::test::queue::IsSyncQueue<Queue>::value)
        {
            BOOST_REQUIRE_EQUAL(
                false,
                alpaka::event::test(e1));

            k1.trigger();

            alpaka::wait::wait(q1);
        }

        BOOST_REQUIRE_EQUAL(
            true,
            alpaka::event::test(e1));
    }
    else
    {
        BOOST_TEST_MESSAGE("Can not execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported!");
    }
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    eventReEnqueueShouldBePossibleIfNobodyWaitsFor,
    TDevQueue,
    TestQueues)
{
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!alpaka::test::queue::IsSyncQueue<Queue>::value)
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
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));

            // q1 = [k1, e1]
            alpaka::queue::enqueue(q1, e1);
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));

            // q1 = [k1, e1, k2]
            alpaka::queue::enqueue(q1, k2);
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k2));

            // re-enqueue should be possible
            // q1 = [k1, k2, e1]
            alpaka::queue::enqueue(q1, e1);
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k2));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));

            // q1 = [k2, e1]
            k1.trigger();
            BOOST_REQUIRE_EQUAL(true, alpaka::event::test(k1));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k2));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));

            // q1 = [e1]
            k2.trigger();
            BOOST_REQUIRE_EQUAL(true, alpaka::event::test(k2));
            alpaka::wait::wait(e1);
            BOOST_REQUIRE_EQUAL(true, alpaka::event::test(e1));
        }
        else
        {
            BOOST_TEST_MESSAGE("Can not execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported!");
        }
    }
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    eventReEnqueueShouldBePossibleIfSomeoneWaitsFor,
    TDevQueue,
    TestQueues)
{
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!alpaka::test::queue::IsSyncQueue<Queue>::value)
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
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));

            // q1 = [k1, e1]
            alpaka::queue::enqueue(q1, e1);
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));

            // q1 = [k1, e1, k2]
            alpaka::queue::enqueue(q1, k2);
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k2));

            // wait for e1
            // q2 = [->e1]
            alpaka::wait::wait(q2, e1);

            // q2 = [->e1, e2]
            alpaka::queue::enqueue(q2, e2);
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e2));

            // re-enqueue should be possible
            // q1 = [k1, e1-old, k2, e1]
            alpaka::queue::enqueue(q1, e1);
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k2));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e2));

            // q1 = [k2, e1]
            k1.trigger();
            BOOST_REQUIRE_EQUAL(true, alpaka::event::test(k1));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k2));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));
            BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e2));

            // q1 = [e1]
            k2.trigger();
            BOOST_REQUIRE_EQUAL(true, alpaka::event::test(k2));
            alpaka::wait::wait(e1);
            BOOST_REQUIRE_EQUAL(true, alpaka::event::test(e1));
            alpaka::wait::wait(e2);
            BOOST_REQUIRE_EQUAL(true, alpaka::event::test(e2));
        }
        else
        {
            BOOST_TEST_MESSAGE("Can not execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported!");
        }
    }
}

//-----------------------------------------------------------------------------
// github issue #388
BOOST_AUTO_TEST_CASE_TEMPLATE(
    waitForEventThatAlreadyFinishedShouldBeSkipped,
    TDevQueue,
    TestQueues)
{
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    using Queue = typename Fixture::Queue;
    using Dev = typename Fixture::Dev;

    if(!alpaka::test::queue::IsSyncQueue<Queue>::value)
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
            BOOST_REQUIRE_EQUAL(true, alpaka::event::test(e1));

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
            BOOST_TEST_MESSAGE("Can not execute test because CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is not supported!");
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
