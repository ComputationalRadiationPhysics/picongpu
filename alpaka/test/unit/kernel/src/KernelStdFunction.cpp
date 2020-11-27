/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#define TEST_UNIT_KERNEL_KERNEL_STD_FUNCTION
// clang thinks a macro is unused when only used as the second operand of an &&
// where the first operand evaluates to false, so we use it here:
TEST_UNIT_KERNEL_KERNEL_STD_FUNCTION

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

#include <functional>
#if BOOST_LANG_CUDA
#    include <nvfunctional>
#endif

//-----------------------------------------------------------------------------
template<typename Acc>
void ALPAKA_FN_ACC kernelFn(Acc const& acc, bool* success, std::int32_t val)
{
    alpaka::ignore_unused(acc);

    ALPAKA_CHECK(*success, 42 == val);
}

// std::function and std::bind is only allowed on CPU
#if !BOOST_LANG_CUDA && !BOOST_LANG_HIP
//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("stdFunctionKernelIsWorking", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    const auto kernel = std::function<void(Acc const&, bool*, std::int32_t)>(kernelFn<Acc>);
    REQUIRE(fixture(kernel, 42));
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("stdBindKernelIsWorking", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    const auto kernel = std::bind(kernelFn<Acc>, std::placeholders::_1, std::placeholders::_2, 42);
    REQUIRE(fixture(kernel));
}
#endif

// This test is disabled due to #836 (cudaErrorIllegalInstruction crash)
#if 0
//#if BOOST_LANG_CUDA
// clang as a native CUDA compiler does not seem to support nvstd::function when ALPAKA_ACC_GPU_CUDA_ONLY_MODE is used.
// error: reference to __device__ function 'kernelFn<alpaka::AccGpuCudaRt<std::__1::integral_constant<unsigned long, 1>, unsigned long> >' in __host__ function const auto kernel = nvstd::function<void(Acc const &, bool *, std::int32_t)>( kernelFn<Acc> );
#    if !(defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE) && BOOST_COMP_CLANG_CUDA)
//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "nvstdFunctionKernelIsWorking", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(
        alpaka::Vec<Dim, Idx>::ones());

    const auto kernel = nvstd::function<void(Acc const &, bool *, std::int32_t)>( kernelFn<Acc> );
    REQUIRE(fixture(kernel, 42));
}

#    endif
#endif

#undef TEST_UNIT_KERNEL_KERNEL_STD_FUNCTION
