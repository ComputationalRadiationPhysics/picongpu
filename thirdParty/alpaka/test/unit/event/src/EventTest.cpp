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
#include <alpaka/test/stream/Stream.hpp>
#include <alpaka/test/stream/StreamTestFixture.hpp>

#include <boost/predef.h>
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
    TDevStream,
    alpaka::test::stream::TestStreams)
{
    using Fixture = alpaka::test::stream::StreamTestFixture<TDevStream>;
    using Stream = typename Fixture::Stream;

    Fixture f;
    alpaka::event::Event<Stream> event(f.m_dev);

    BOOST_REQUIRE_EQUAL(
        true,
        alpaka::event::test(event));
}

// All of the following tests use the EventHostManualTrigger which is only available on CUDA 8.0+
#if !BOOST_LANG_CUDA || BOOST_LANG_CUDA >= BOOST_VERSION_NUMBER(8, 0, 0)
using TestStreams = alpaka::test::stream::TestStreams;
#else
using TestStreams = alpaka::test::stream::TestStreamsCpu;
#endif

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    eventTestShouldBeFalseWhileInQueueAndTrueAfterBeingProcessed,
    TDevStream,
    TestStreams)
{
    using Fixture = alpaka::test::stream::StreamTestFixture<TDevStream>;
    using Stream = typename Fixture::Stream;
    using Dev = typename Fixture::Dev;

    Fixture f1;
    auto s1 = f1.m_stream;
    alpaka::event::Event<Stream> e1(f1.m_dev);
    alpaka::test::event::EventHostManualTrigger<Dev> k1(f1.m_dev);

    if(!alpaka::test::stream::IsSyncStream<Stream>::value)
    {
        alpaka::stream::enqueue(s1, k1);
    }

    alpaka::stream::enqueue(s1, e1);

    if(!alpaka::test::stream::IsSyncStream<Stream>::value)
    {
        BOOST_REQUIRE_EQUAL(
            false,
            alpaka::event::test(e1));

        k1.trigger();

        alpaka::wait::wait(s1);
    }

    BOOST_REQUIRE_EQUAL(
        true,
        alpaka::event::test(e1));
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    eventReEnqueueShouldBePossibleIfNobodyWaitsFor,
    TDevStream,
    TestStreams)
{
    using Fixture = alpaka::test::stream::StreamTestFixture<TDevStream>;
    using Stream = typename Fixture::Stream;
    using Dev = typename Fixture::Dev;

    if(!alpaka::test::stream::IsSyncStream<Stream>::value)
    {
        Fixture f1;
        auto s1 = f1.m_stream;
        alpaka::event::Event<Stream> e1(f1.m_dev);
        alpaka::test::event::EventHostManualTrigger<Dev> k1(f1.m_dev);
        alpaka::test::event::EventHostManualTrigger<Dev> k2(f1.m_dev);

        // s1 = [k1]
        alpaka::stream::enqueue(s1, k1);
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));

        // s1 = [k1, e1]
        alpaka::stream::enqueue(s1, e1);
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));

        // s1 = [k1, e1, k2]
        alpaka::stream::enqueue(s1, k2);
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k2));

        // re-enqueue should be possible
        // s1 = [k1, k2, e1]
        alpaka::stream::enqueue(s1, e1);
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k2));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));

        // s1 = [k2, e1]
        k1.trigger();
        BOOST_REQUIRE_EQUAL(true, alpaka::event::test(k1));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k2));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));

        // s1 = [e1]
        k2.trigger();
        BOOST_REQUIRE_EQUAL(true, alpaka::event::test(k2));
        alpaka::wait::wait(e1);
        BOOST_REQUIRE_EQUAL(true, alpaka::event::test(e1));
    }
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    eventReEnqueueShouldBePossibleIfSomeoneWaitsFor,
    TDevStream,
    TestStreams)
{
    using Fixture = alpaka::test::stream::StreamTestFixture<TDevStream>;
    using Stream = typename Fixture::Stream;
    using Dev = typename Fixture::Dev;

    if(!alpaka::test::stream::IsSyncStream<Stream>::value)
    {
        Fixture f1;
        Fixture f2;
        auto s1 = f1.m_stream;
        auto s2 = f2.m_stream;
        alpaka::event::Event<Stream> e1(f1.m_dev);
        alpaka::event::Event<Stream> e2(f2.m_dev);
        alpaka::test::event::EventHostManualTrigger<Dev> k1(f1.m_dev);
        alpaka::test::event::EventHostManualTrigger<Dev> k2(f1.m_dev);

        // s1 = [k1]
        alpaka::stream::enqueue(s1, k1);
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));

        // s1 = [k1, e1]
        alpaka::stream::enqueue(s1, e1);
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));

        // s1 = [k1, e1, k2]
        alpaka::stream::enqueue(s1, k2);
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k2));

        // wait for e1
        // s2 = [->e1]
        alpaka::wait::wait(s2, e1);

        // s2 = [->e1, e2]
        alpaka::stream::enqueue(s2, e2);
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e2));

        // re-enqueue should be possible
        // s1 = [k1, k2, e1]
        alpaka::stream::enqueue(s1, e1);
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k1));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k2));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e2));

        // s1 = [k2, e1]
        k1.trigger();
        BOOST_REQUIRE_EQUAL(true, alpaka::event::test(k1));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(k2));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e1));
        BOOST_REQUIRE_EQUAL(false, alpaka::event::test(e2));

        // s1 = [e1]
        k2.trigger();
        BOOST_REQUIRE_EQUAL(true, alpaka::event::test(k2));
        alpaka::wait::wait(e1);
        BOOST_REQUIRE_EQUAL(true, alpaka::event::test(e1));
        alpaka::wait::wait(e2);
        BOOST_REQUIRE_EQUAL(true, alpaka::event::test(e2));
    }
}

//-----------------------------------------------------------------------------
// github issue #388
BOOST_AUTO_TEST_CASE_TEMPLATE(
    waitForEventThatAlreadyFinishedShouldBeSkipped,
    TDevStream,
    TestStreams)
{
    using Fixture = alpaka::test::stream::StreamTestFixture<TDevStream>;
    using Stream = typename Fixture::Stream;
    using Dev = typename Fixture::Dev;

    if(!alpaka::test::stream::IsSyncStream<Stream>::value)
    {
        Fixture f1;
        Fixture f2;
        auto s1 = f1.m_stream;
        auto s2 = f2.m_stream;
        alpaka::test::event::EventHostManualTrigger<Dev> k1(f1.m_dev);
        alpaka::test::event::EventHostManualTrigger<Dev> k2(f2.m_dev);
        alpaka::event::Event<Stream> e1(f1.m_dev);

        // 1. kernel k1 is enqueued into stream s1
        // s1 = [k1]
        alpaka::stream::enqueue(s1, k1);
        // 2. kernel k2 is enqueued into stream s2
        // s2 = [k2]
        alpaka::stream::enqueue(s2, k2);

        // 3. event e1 is enqueued into stream s1
        // s1 = [k1, e1]
        alpaka::stream::enqueue(s1, e1);

        // 4. s2 waits for e1
        // s2 = [k2, ->e1]
        alpaka::wait::wait(s2, e1);

        // 5. kernel k1 finishes
        // s1 = [e1]
        k1.trigger();

        // 6. e1 is finished
        // s1 = []
        alpaka::wait::wait(e1);
        BOOST_REQUIRE_EQUAL(true, alpaka::event::test(e1));

        // 7. e1 is re-enqueued again but this time into s2
        // s2 = [k2, ->e1, e1]
        alpaka::stream::enqueue(s2, e1);

        // 8. kernel k2 finishes
        // s2 = [->e1, e1]
        k2.trigger();

        // 9. e1 had already been signaled so there should not be waited even though the event is now reused within s2 and its current state is 'unfinished' again.
        // s2 = [e1]

        // Both streams should successfully finish
        alpaka::wait::wait(s1);
        // s2 = []
        alpaka::wait::wait(s2);
    }
}

BOOST_AUTO_TEST_SUITE_END()
