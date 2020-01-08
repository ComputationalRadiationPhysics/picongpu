/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/kernel/Traits.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/core/BoostPredef.hpp>

#include <catch2/catch.hpp>

#include <functional>
#if BOOST_LANG_CUDA
#include <nvfunctional>
#endif

//-----------------------------------------------------------------------------
template<
    typename TAcc>
void ALPAKA_FN_ACC kernelFn(
    TAcc const & acc,
    bool * success,
    std::int32_t val)
{
    alpaka::ignore_unused(acc);

    ALPAKA_CHECK(*success, 42 == val);
}

// std::function and std::bind is only allowed on CPU
#if !BOOST_LANG_CUDA && !BOOST_LANG_HIP
//-----------------------------------------------------------------------------
struct TestTemplateStdFunction
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    const auto kernel = std::function<void(TAcc const &, bool *, std::int32_t)>( kernelFn<TAcc> );
    REQUIRE(fixture(kernel, 42));
  }
};

TEST_CASE( "stdFunctionKernelIsWorking", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateStdFunction() );
}

//-----------------------------------------------------------------------------
struct TestTemplateStdBind
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    const auto kernel = std::bind( kernelFn<TAcc>, std::placeholders::_1, std::placeholders::_2, 42 );
    REQUIRE(fixture(kernel));
  }
};

TEST_CASE( "stdBindKernelIsWorking", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateStdBind() );
}
#endif

// This test is disabled due to #836 (cudaErrorIllegalInstruction crash)
#if 0
//#if BOOST_LANG_CUDA
// clang as a native CUDA compiler does not seem to support nvstd::function when ALPAKA_ACC_GPU_CUDA_ONLY_MODE is used.
// error: reference to __device__ function 'kernelFn<alpaka::acc::AccGpuCudaRt<std::__1::integral_constant<unsigned long, 1>, unsigned long> >' in __host__ function const auto kernel = nvstd::function<void(TAcc const &, bool *, std::int32_t)>( kernelFn<TAcc> );
#if !(defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE) && BOOST_COMP_CLANG_CUDA)
//-----------------------------------------------------------------------------
struct TestTemplateNvstdFunction
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    const auto kernel = nvstd::function<void(TAcc const &, bool *, std::int32_t)>( kernelFn<TAcc> );
    REQUIRE(fixture(kernel, 42));
  }
};

TEST_CASE( "nvstdFunctionKernelIsWorking", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateNvstdFunction() );
}
#endif
#endif
