/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#if _OPENMP < 200203
    #error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#endif

#include <alpaka/alpaka.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/queue/QueueTestFixture.hpp>
#include <alpaka/test/queue/QueueCpuOmp2Collective.hpp>

#include <vector>

#include <catch2/catch.hpp>

struct QueueCollectiveTestKernel
{
    template<typename TAcc>
    auto operator()(
        TAcc const & acc,
        int* resultsPtr) const
    -> void
    {
        size_t threadId = alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
        // avoid that one thread is doing all the work
        std::this_thread::sleep_for(std::chrono::milliseconds(200u * threadId));
        resultsPtr[threadId] = static_cast<int>(threadId);
    }
};

TEST_CASE("queueCollective", "[queue]")
{
    // Define the index domain
    using Dim = alpaka::dim::DimInt<1>;
    using Idx = size_t;

    // Define the accelerator
    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Idx>;
    using Dev = alpaka::dev::Dev<Acc>;

    using Queue = alpaka::queue::QueueCpuOmp2Collective;
    using Pltf = alpaka::pltf::Pltf<Dev>;

    auto dev = alpaka::pltf::getDevByIdx<Pltf>(0u);
    Queue queue(dev);

    std::vector<int> results(4, -1);

    using Vec = alpaka::vec::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
    Vec const threadsPerBlock(Vec::all(static_cast<Idx>(1)));
    Vec const blocksPerGrid(results.size());

    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
    WorkDiv const workDiv(
        blocksPerGrid,
        threadsPerBlock,
        elementsPerThread);

    #pragma omp parallel num_threads(static_cast<int>(results.size()))
    {
        // The kernel will be performed collectively.
        // OpenMP will distribute the work between the threads from the parallel region
        alpaka::kernel::exec<Acc>(
               queue,
               workDiv,
               QueueCollectiveTestKernel{},
               results.data());

        alpaka::wait::wait(queue);
    }

    for(size_t i = 0; i < results.size(); ++i)
    {
        REQUIRE(static_cast<int>(i) == results.at(i));
    }
}

TEST_CASE("TestCollectiveMemcpy", "[queue]")
{
     // Define the index domain
    using Dim = alpaka::dim::DimInt<1>;
    using Idx = size_t;

    // Define the accelerator
    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Idx>;
    using Dev = alpaka::dev::Dev<Acc>;

    using Queue = alpaka::queue::QueueCpuOmp2Collective;
    using Pltf = alpaka::pltf::Pltf<Dev>;

    auto dev = alpaka::pltf::getDevByIdx<Pltf>(0u);
    Queue queue(dev);

    std::vector<int> results(4, -1);

    // Define the work division
    using Vec = alpaka::vec::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
    Vec const threadsPerBlock(Vec::all(static_cast<Idx>(1)));
    Vec const blocksPerGrid(results.size());

    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
    WorkDiv const workDiv(
        blocksPerGrid,
        threadsPerBlock,
        elementsPerThread);

    #pragma omp parallel num_threads(static_cast<int>(results.size()))
    {
        int threadId = omp_get_thread_num();

        using View = alpaka::mem::view::ViewPlainPtr<Dev, int, Dim, Idx>;

        View dst(
            results.data() + threadId,
            dev,
            Vec(static_cast<Idx>(1u)),
            Vec(sizeof(int)));

        View src(
            &threadId,
            dev,
            Vec(static_cast<Idx>(1u)),
            Vec(sizeof(int)));

        // avoid that the first thread is executing the copy (can not be guaranteed)
        size_t sleep_ms = (results.size() - static_cast<uint32_t>(threadId)) * 100u;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));

        // only one thread will perform this memcpy
        alpaka::mem::view::copy(queue, dst, src, Vec(static_cast<Idx>(1u)));

        alpaka::wait::wait(queue);
    }

    uint32_t numFlippedValues = 0u;
    uint32_t numNonIntitialValues = 0u;
    for(size_t i = 0; i < results.size(); ++i)
    {
        if(static_cast<int>(i) == results.at(i))
            numFlippedValues++;
        if(results.at(i) != -1)
            numNonIntitialValues++;
    }
    // only one thread is allowed to flip the value
    REQUIRE(numFlippedValues == 1u);
    // only one value is allowed to differ from the initial value
    REQUIRE(numNonIntitialValues == 1u);
}

#endif
