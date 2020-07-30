/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/kernel/Traits.hpp>

#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch.hpp>

//#############################################################################
//! It is not possible to use a alpaka kernel function object without a templated operator() when the CUDA accelerator is hard-coded.
//!
//! However, compiling such kernels with a CPU device works fine.
//!
//! When the CUDA accelerator is used, the following error is triggered:
//! /alpaka/include/alpaka/workdiv/Traits.hpp(...): error: calling a __device__ function("getWorkDiv") from a __host__ __device__ function("getWorkDiv") is not allowed
//! The kernel function objects function call operator is attributed with ALPAKA_FN_ACC which is identical to __host__ __device__.
//! The 'alpaka::workdiv::getWorkDiv<...>(acc)' function that is called has the ALPAKA_FN_HOST_ACC attribute (also equal to __host__ __device__).
//! The underlying trait calls the CUDA specialized method which has the __device__ attribute.
//! Because this call chain does not contain any templates and therefore no calls depending on input types,
//! everything can be resolved at the first time the template is parsed which results in the given error.
//!
//! Currently, the only possible way to solve this is to make the function call operator a template nonetheless by providing an unused template parameter.

using Dim = alpaka::dim::DimInt<2u>;
using Idx = std::uint32_t;
#if defined(ALPAKA_ACC_CPU_SERIAL_ENABLED)
using AccCpu = alpaka::acc::AccCpuSerial<Dim, Idx>;
#endif
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP
using AccGpu = alpaka::acc::AccGpuHipRt<Dim, Idx>;
#elif defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
using AccGpu = alpaka::acc::AccGpuCudaRt<Dim, Idx>;
#endif

#if defined(ALPAKA_ACC_CPU_SERIAL_ENABLED)
//#############################################################################
struct KernelNoTemplateCpu
{
    //-----------------------------------------------------------------------------
    ALPAKA_FN_ACC
    auto operator()(
        AccCpu const & acc,
        bool* success) const
    -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::idx::Idx<AccCpu>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());
    }
};

//-----------------------------------------------------------------------------
TEST_CASE("kernelNoTemplateCpu", "[kernel]")
{
    alpaka::test::KernelExecutionFixture<AccCpu> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelNoTemplateCpu kernel;

    REQUIRE(fixture(kernel));
}
#endif

/*#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
//#############################################################################
//! DO NOT ENABLE! COMPILATION WILL FAIL!
struct KernelNoTemplateGpu
{
    //-----------------------------------------------------------------------------
    ALPAKA_FN_ACC
    auto operator()(
        AccGpu const & acc,
        bool* success) const
    -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::idx::Idx<AccGpu>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());
    }
};

//-----------------------------------------------------------------------------
TEST_CASE("kernelNoTemplateGpu", "[kernel]")
{
    alpaka::test::KernelExecutionFixture<AccGpu> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelNoTemplateGpu kernel;

    REQUIRE(fixture(kernel));
}
#endif*/

#if defined(ALPAKA_ACC_CPU_SERIAL_ENABLED)
//#############################################################################
struct KernelWithoutTemplateParamCpu
{
    //-----------------------------------------------------------------------------
    template<
        typename TNotUsed = void>
    ALPAKA_FN_ACC
    auto operator()(
        AccCpu const & acc,
        bool* success) const
    -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::idx::Idx<AccCpu>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());
    }
};

//-----------------------------------------------------------------------------
TEST_CASE("kernelWithoutTemplateParamCpu", "[kernel]")
{
    alpaka::test::KernelExecutionFixture<AccCpu> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelWithoutTemplateParamCpu kernel;

    REQUIRE(fixture(kernel));
}
#endif

#if (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) \
  || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
//#############################################################################
struct KernelWithoutTemplateParamGpu
{
    //-----------------------------------------------------------------------------
    template<
        typename TNotUsed = void>
    ALPAKA_FN_ACC
    auto operator()(
        AccGpu const & acc,
        bool* success) const
    -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::idx::Idx<AccGpu>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());
    }
};

//-----------------------------------------------------------------------------
TEST_CASE("kernelWithoutTemplateParamGpu", "[kernel]")
{
    alpaka::test::KernelExecutionFixture<AccGpu> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelWithoutTemplateParamGpu kernel;

    REQUIRE(fixture(kernel));
}
#endif
