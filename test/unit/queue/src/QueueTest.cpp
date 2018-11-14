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

#include <future>
#include <thread>

BOOST_AUTO_TEST_SUITE(queue)


//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    queueIsInitiallyEmpty,
    TDevQueue,
    alpaka::test::queue::TestQueues)
{
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    Fixture f;

    BOOST_CHECK_EQUAL(true, alpaka::queue::empty(f.m_queue));
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    queueCallbackIsWorking,
    TDevQueue,
    alpaka::test::queue::TestQueues)
{
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    Fixture f;

    std::promise<bool> promise;

    alpaka::queue::enqueue(
        f.m_queue,
        [&](){
            promise.set_value(true);
        }
    );

    BOOST_CHECK_EQUAL(true, promise.get_future().get());
#endif
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    queueWaitShouldWork,
    TDevQueue,
    alpaka::test::queue::TestQueues)
{
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    Fixture f;

    bool CallbackFinished = false;
    alpaka::queue::enqueue(
        f.m_queue,
        [&CallbackFinished]() noexcept
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100u));
            CallbackFinished = true;
        });

    alpaka::wait::wait(f.m_queue);
    BOOST_CHECK_EQUAL(true, CallbackFinished);
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    queueShouldNotBeEmptyWhenLastTaskIsStillExecutingAndIsEmptyAfterProcessingFinished,
    TDevQueue,
    alpaka::test::queue::TestQueues)
{
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    Fixture f;

    bool CallbackFinished = false;
    alpaka::queue::enqueue(
        f.m_queue,
        [&f, &CallbackFinished]() noexcept
        {
            BOOST_CHECK_EQUAL(false, alpaka::queue::empty(f.m_queue));
            std::this_thread::sleep_for(std::chrono::milliseconds(100u));
            CallbackFinished = true;
        });

    // A synchronous queue will always stay empty because the task has been executed immediately.
    if(!alpaka::test::queue::IsSyncQueue<typename Fixture::Queue>::value)
    {
        alpaka::wait::wait(f.m_queue);
    }

    BOOST_CHECK_EQUAL(true, alpaka::queue::empty(f.m_queue));
    BOOST_CHECK_EQUAL(true, CallbackFinished);
}

BOOST_AUTO_TEST_SUITE_END()
