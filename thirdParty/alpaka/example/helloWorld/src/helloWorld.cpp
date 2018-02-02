/**
 * \file
 * Copyright 2014-2015 Erik Zenker
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
        // In the most cases the parallel work distibution depends
        // on the current index of a thread and how many threads
        // exist overall. These information can be obtained by
        // getIdx() and getWorkDiv(). In this example these
        // values are obtained for a global scope.
        auto const globalThreadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        // Map the three dimensional thread index into a
        // one dimensional thread index space. We call it
        // linearize the thread index.
        auto const linearizedGlobalThreadIdx = alpaka::idx::mapIdx<1u>(
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
    // Define accelerator types
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
    using Dim = alpaka::dim::DimInt<3>;
    using Size = std::size_t;
    using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Stream = alpaka::stream::StreamCpuSync;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using DevHost = alpaka::dev::Dev<Host>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;


    // Get the first devices
    //
    // The accelerator only defines how something should be
    // parallized, but a device is the real entity which will
    // run the parallel programm. The device can be choosen
    // by id (0 to the number of devices minus 1) or you
    // can also retrieve all devices in a vector (getDevs()).
    // In this example the first devices is choosen.
    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // Create a stream to the accelerator device
    //
    // A stream can be interpreted as the work queue
    // of a particular device. Streams are filled with
    // executors and alpaka takes care that these
    // executors will be executed. Streams are provided in
    // async and sync variants.
    // The example stream is a sync stream to a cpu device,
    // but it also exists an async stream for this
    // device (StreamCpuAsync).
    Stream stream(devAcc);

    // Init workdiv
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
    alpaka::vec::Vec<Dim, Size> const elementsPerThread(
        static_cast<Size>(1),
        static_cast<Size>(1),
        static_cast<Size>(1));

    alpaka::vec::Vec<Dim, Size> const threadsPerBlock(
        static_cast<Size>(1),
        static_cast<Size>(1),
        static_cast<Size>(1));

    alpaka::vec::Vec<Dim, Size> const blocksPerGrid(
        static_cast<Size>(4),
        static_cast<Size>(8),
        static_cast<Size>(16));

    WorkDiv const workdiv(
        blocksPerGrid,
        threadsPerBlock,
        elementsPerThread);


    // Run kernel
    //
    // Kernels need to be provided as classes or structs
    // which provide a public operator(). This operator is
    // the actual method that should be accelerated. An
    // object of the kernel is used to create an execution
    // unit and this unit is finally enqueued into an
    // accelerator stream. The enqueuing can be done
    // synchronously or asynchronously depending on the choosen
    // stream (see type definitions above).
    HelloWorldKernel helloWorldKernel;

    auto const helloWorld(alpaka::exec::create<Acc>(
        workdiv,
        helloWorldKernel
        /* put kernel arguments here */));

    alpaka::stream::enqueue(stream, helloWorld);

    return EXIT_SUCCESS;
}
