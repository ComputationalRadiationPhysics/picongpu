/**
 * \file
 * Copyright 2014-2018 Erik Zenker, Benjamin Worpitz
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
 *
 */

#include <alpaka/alpaka.hpp>

#include <iostream>

//#############################################################################
//! Hello World Kernel
//!
//! Prints "[x, y, z][gtid] Hello World" where tid is the global thread number.
struct HelloWorldKernel
{
    //-----------------------------------------------------------------------------
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc) const
    -> void
    {
        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;
        using Vec = alpaka::vec::Vec<Dim, Idx>;
        using Vec1 = alpaka::vec::Vec<alpaka::dim::DimInt<1u>, Idx>;

        // In the most cases the parallel work distibution depends
        // on the current index of a thread and how many threads
        // exist overall. These information can be obtained by
        // getIdx() and getWorkDiv(). In this example these
        // values are obtained for a global scope.
        Vec const globalThreadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        Vec const globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        // Map the three dimensional thread index into a
        // one dimensional thread index space. We call it
        // linearize the thread index.
        Vec1 const linearizedGlobalThreadIdx = alpaka::idx::mapIdx<1u>(
            globalThreadIdx,
            globalThreadExtent);

        // Each thread prints a hello world to the terminal
        // together with the global index of the thread in
        // each dimension and the linearized global index.
        // Mind, that alpaka uses the mathematical index
        // order [z][y][x] where the last index is the fast one.
        printf(
            "[z:%u, y:%u, x:%u][linear:%u] Hello World\n",
            static_cast<unsigned>(globalThreadIdx[0u]),
            static_cast<unsigned>(globalThreadIdx[1u]),
            static_cast<unsigned>(globalThreadIdx[2u]),
            static_cast<unsigned>(linearizedGlobalThreadIdx[0u]));
    }
};

auto main()
-> int
{
// This example is hard-coded to use the sequential executor.
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

    // Define the index domain
    //
    // Depending on your type of problem, you have to define
    // the dimensionality as well as the type used for indices.
    // For small index domains 16 or 32 bit indices may be enough
    // and may be faster to calculate depending on the accelerator.
    using Dim = alpaka::dim::DimInt<3>;
    using Idx = std::size_t;

    // Define the accelerator
    //
    // It is possible to choose from a set of accelerators
    // that are defined in the alpaka::acc namespace e.g.:
    // - AccGpuCudaRt
    // - AccCpuThreads
    // - AccCpuFibers
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccCpuOmp4
    // - AccCpuSerial
    //
    // Each accelerator has strengths and weaknesses. Therefore,
    // they need to be choosen carefully depending on the actual
    // use case. Furthermore, some accelerators only support a
    // particular workdiv, but workdiv can also be generated
    // automatically.

    // By exchanging the Acc and Queue types you can select where to execute the kernel.
#if 1
    using Acc = alpaka::acc::AccCpuSerial<Dim, Idx>;
    using Queue = alpaka::queue::QueueCpuSync;
#else
    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Idx>;
    using Queue = alpaka::queue::QueueCudaRtSync;
#endif
    using Dev = alpaka::dev::Dev<Acc>;
    using Pltf = alpaka::pltf::Pltf<Dev>;


    // Select a device
    //
    // The accelerator only defines how something should be
    // parallized, but a device is the real entity which will
    // run the parallel programm. The device can be choosen
    // by id (0 to the number of devices minus 1) or you
    // can also retrieve all devices in a vector (getDevs()).
    // In this example the first devices is choosen.
    Dev const devAcc(alpaka::pltf::getDevByIdx<Pltf>(0u));

    // Create a queue on the device
    //
    // A queue can be interpreted as the work queue
    // of a particular device. Queues are filled with
    // executors and alpaka takes care that these
    // executors will be executed. Queues are provided in
    // async and sync variants.
    // The example queue is a sync queue to a cpu device,
    // but it also exists an async queue for this
    // device (QueueCpuAsync).
    Queue queue(devAcc);

    // Define the work division
    //
    // A kernel is executed for each element of a
    // n-dimensional grid distinguished by the element indices.
    // The work division defines the number of kernel instantiations as
    // well as the type of parallelism used by the executor.
    // Different accelerators have different requirements on the work
    // division. For example, the sequential accelerator can not
    // provide any thread level parallelism (synchronizable as well as non synchronizable),
    // whereas the CUDA accelerator can spawn hundreds of synchronizing
    // and non synchronizing threads at the same time.
    //
    // The workdiv is divided in three levels of parallelization:
    // - grid-blocks:      The number of blocks in the grid (parallel, not synchronizable)
    // - block-threads:    The number of threads per block (parallel, synchronizable).
    //                     Each thread executes one kernel invocation.
    // - thread-elements:  The number of elements per thread (sequential, not synchronizable).
    //                     Each kernel has to execute its elements sequentially.
    //
    // - Grid     : consists of blocks
    // - Block    : consists of threads
    // - Elements : consists of elements
    //
    // Threads in the same grid can access the same global memory,
    // while threads in the same block can access the same shared
    // memory. Elements are supposed to be used for vectorization.
    // Thus, a thread can process data element size wise with its
    // vector processing unit.
    using Vec = alpaka::vec::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
    Vec const threadsPerBlock(Vec::all(static_cast<Idx>(1)));
    Vec const blocksPerGrid(
        static_cast<Idx>(4),
        static_cast<Idx>(8),
        static_cast<Idx>(16));

    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
    WorkDiv const workDiv(
        blocksPerGrid,
        threadsPerBlock,
        elementsPerThread);


    // Instantiate the kernel function object
    //
    // Kernels can be everything that has a callable operator()
    // and which takes the accelerator as first argument.
    // So a kernel can be a class or struct, a lambda, a std::function, etc.
    HelloWorldKernel helloWorldKernel;

    // Run the kernel
    //
    // To execute the kernel, you have to provide the
    // work division as well as the additional kernel function
    // parameters.
    // The kernel execution task is enqueued into an accelerator queue.
    // The queue can be synchronously or asynchronously
    // depending on the choosen queue type (see type definitions above).
    // Here it is synchronous which means that the kernel is directly executed.
    alpaka::kernel::exec<Acc>(
        queue,
        workDiv,
        helloWorldKernel
        /* put kernel arguments here */);

    return EXIT_SUCCESS;

#else
    return EXIT_SUCCESS;
#endif
}
