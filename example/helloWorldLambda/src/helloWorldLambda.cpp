/**
 * \file
 * Copyright 2014-2016 Erik Zenker
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
template<typename Acc>
void ALPAKA_FN_ACC hiWorldFunction(Acc& acc, size_t const nExclamationMarks){
    auto globalThreadIdx    = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    auto globalThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
    auto linearizedGlobalThreadIdx = alpaka::idx::mapIdx<1u>(globalThreadIdx,
                                                              globalThreadExtent);
                                                          
    printf("[z:%u, y:%u, x:%u][linear:%u] Hi world from a std::function",
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
    // Define accelerator types
    using Dim = alpaka::dim::DimInt<3>;
    using Size = std::size_t;
    using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
    //using Acc = alpaka::acc::AccGpuCudaRt<Dim, Size>;    
    using Stream = alpaka::stream::StreamCpuSync;
    //using Stream = alpaka::stream::StreamCudaRtSync;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using DevHost = alpaka::dev::Dev<Host>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;

    // Get the first devices
    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // Create a stream to the accelerator device
    Stream stream(devAcc);

    // Init workdiv
    alpaka::vec::Vec<Dim, Size> const elementsPerThread(
        static_cast<Size>(1),
        static_cast<Size>(1),
        static_cast<Size>(1));

    alpaka::vec::Vec<Dim, Size> const threadsPerBlock(
        static_cast<Size>(1),
        static_cast<Size>(1),
        static_cast<Size>(1));

    alpaka::vec::Vec<Dim, Size> const blocksPerGrid(
        static_cast<Size>(1),
        static_cast<Size>(2),
        static_cast<Size>(4));

    WorkDiv const workdiv(
        blocksPerGrid,
        threadsPerBlock,
        elementsPerThread);

    const size_t nExclamationMarks = 10;

// nvcc < 7.5 does not support lambdas as kernels.
#if !BOOST_COMP_NVCC || BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(7, 5, 0)
// nvcc 7.5 does not support heterogeneous lambdas (__host__ __device__) as kernels but only __device__ lambdas.
// So with nvcc 7.5 this only works in CUDA only mode or by using ALPAKA_FN_ACC_CUDA_ONLY instead of ALPAKA_FN_ACC
#if !BOOST_COMP_NVCC || BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(8, 0, 0) || defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE)

// clang prior to 4.0.0 did not support the __host__ __device__ attributes at the nonstandard position between [] and () but only after ().
// See: https://llvm.org/bugs/show_bug.cgi?id=26341
#if !BOOST_COMP_CLANG_CUDA || BOOST_COMP_CLANG_CUDA >= BOOST_VERSION_NUMBER(4, 0, 0)

    // Run "Hello World" kernel with lambda function
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
    // 
    // This kind of kernel function
    // declaration might be useful when small kernels
    // are written for testing or lambda functions
    // already exist.
    auto const helloWorld(alpaka::exec::create<Acc>(
        workdiv,
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
    ));

    alpaka::stream::enqueue(stream, helloWorld);

#endif
#endif
#endif
    
    // Run "Hello World" kernel with std::function
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
    auto const hiWorld (alpaka::exec::create<Acc> (
        workdiv,
        std::function<void(Acc&, size_t)>( hiWorldFunction<Acc> ),
        nExclamationMarks));

    alpaka::stream::enqueue(stream, hiWorld);

    
    // Run "Hello World" kernel with std::bind
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
    auto const hiWorldBind (alpaka::exec::create<Acc> (
        workdiv,
        std::bind( hiWorldFunction<Acc>, std::placeholders::_1, nExclamationMarks*2 )
        ));

    alpaka::stream::enqueue(stream, hiWorldBind);

    return EXIT_SUCCESS;
}
