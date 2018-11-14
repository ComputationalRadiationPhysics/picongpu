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

#include <functional>

//-----------------------------------------------------------------------------
//! This functions says hi to the world and
//! can be encapsulated into a std::function
//! and used as a kernel function. It is 
//! just another way to define alpaka kernels
//! and might be useful when it is necessary
//! to lift an existing function into a kernel
//! function.
template<
    typename TAcc>
void ALPAKA_FN_ACC hiWorldFunction(
    TAcc& acc,
    size_t const nExclamationMarks)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;
    using Vec = alpaka::vec::Vec<Dim, Idx>;
    using Vec1 = alpaka::vec::Vec<alpaka::dim::DimInt<1u>, Idx>;

    Vec const globalThreadIdx    = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    Vec const globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
    Vec1 const linearizedGlobalThreadIdx = alpaka::idx::mapIdx<1u>(globalThreadIdx,
                                                              globalThreadExtent);

    printf("[z:%u, y:%u, x:%u][linear:%u] Hi world from a function",
           static_cast<unsigned>(globalThreadIdx[0]),
           static_cast<unsigned>(globalThreadIdx[1]),
           static_cast<unsigned>(globalThreadIdx[2]),
           static_cast<unsigned>(linearizedGlobalThreadIdx[0]));

    for(size_t i = 0; i < nExclamationMarks; ++i){
        printf("!");
    }
                                                          
    printf("\n");  
}

auto main()
-> int
{
// This example is hard-coded to use the sequential executor.
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

    // Define the index domain
    using Dim = alpaka::dim::DimInt<3>;
    using Idx = std::size_t;

    // Define the accelerator
    using Acc = alpaka::acc::AccCpuSerial<Dim, Idx>;
    using Queue = alpaka::queue::QueueCpuSync;
    using Dev = alpaka::dev::Dev<Acc>;
    using Pltf = alpaka::pltf::Pltf<Dev>;

    // Select a device
    Dev const devAcc(alpaka::pltf::getDevByIdx<Pltf>(0u));

    // Create a queue on the device
    Queue queue(devAcc);

    // Define the work division
    using Vec = alpaka::vec::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
    Vec const threadsPerBlock(Vec::all(static_cast<Idx>(1)));
    Vec const blocksPerGrid(
        static_cast<Idx>(1),
        static_cast<Idx>(2),
        static_cast<Idx>(4));

    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
    WorkDiv const workDiv(
        blocksPerGrid,
        threadsPerBlock,
        elementsPerThread);

    const size_t nExclamationMarks = 10;

    // Run "Hello World" kernel with a lambda function
    //
    // Alpaka is able to execute lambda functions (anonymous functions) which
    // are available since the C++11 standard.
    // Alpaka forces the lambda function to accept
    // the utilized accelerator as first argument. 
    // All following arguments can be provided after
    // the lambda function declaration or be captured. 
    //
    // This example passes the number exclamation marks, that should
    // be written after we greet the world, to the 
    // lambda function.
    alpaka::kernel::exec<Acc>(
        queue,
        workDiv,
        [] ALPAKA_FN_ACC (Acc & acc, size_t const nExclamationMarksAsArg) -> void {
            auto globalThreadIdx    = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            auto globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
            auto linearizedGlobalThreadIdx = alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

            printf("[z:%u, y:%u, x:%u][linear:%u] Hello world from a lambda",
               static_cast<unsigned>(globalThreadIdx[0]),
               static_cast<unsigned>(globalThreadIdx[1]),
               static_cast<unsigned>(globalThreadIdx[2]),
               static_cast<unsigned>(linearizedGlobalThreadIdx[0]));

            for(size_t i = 0; i < nExclamationMarksAsArg; ++i){
                printf("!");
            }

            printf("\n");

        },
        nExclamationMarks
    );

    
    // Run "Hello World" kernel with a std::function
    //
    // This kernel says hi to world by using 
    // std::functions, which are available since
    // the C++11 standard.
    // The interface for std::function can be used
    // to encapsulate normal c++ functions and 
    // lambda functions into a function object. 
    // Alpaka accepts these std::functions 
    // as kernel functions. Therefore, it is easy
    // to wrap allready existing code into a
    // std::function and provide it to the alpaka 
    // library.
    alpaka::kernel::exec<Acc> (
        queue,
        workDiv,
        std::function<void(Acc&, size_t)>( hiWorldFunction<Acc> ),
        nExclamationMarks);

    
    // Run "Hello World" kernel with a std::bind function object
    //
    // This kernel binds arguments of the existing function hiWorldFunction
    // to a std::function objects and provides it as alpaka kernel.
    // The syntax needs to be the following:
    // - std::bind(foo<Acc>, std::placeholders::_1, arg1, arg2, ...)
    //
    // The placeholder will be filled by alpaka with the 
    // particular accelerator object.
    //
    // This approach has the advantage that you do
    // not need to provide the signature of your function
    // as it is the case for the std::function example above.
    alpaka::kernel::exec<Acc> (
        queue,
        workDiv,
        std::bind( hiWorldFunction<Acc>, std::placeholders::_1, nExclamationMarks*2 )
        );

    return EXIT_SUCCESS;

#else
    return EXIT_SUCCESS;
#endif
}
