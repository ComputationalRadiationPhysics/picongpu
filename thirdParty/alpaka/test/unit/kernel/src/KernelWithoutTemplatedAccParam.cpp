/* Copyright 2024 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/kernel/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch_test_macros.hpp>

using Dim = alpaka::DimInt<2u>;
using Idx = std::uint32_t;
#if defined(ALPAKA_ACC_CPU_SERIAL_ENABLED)
using AccCpu = alpaka::AccCpuSerial<Dim, Idx>;
#endif
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP
using AccGpu = alpaka::AccGpuHipRt<Dim, Idx>;
#elif defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
using AccGpu = alpaka::AccGpuCudaRt<Dim, Idx>;
#endif

#if defined(ALPAKA_ACC_CPU_SERIAL_ENABLED)
struct KernelNoTemplateCpu
{
    ALPAKA_FN_ACC auto operator()(AccCpu const& acc, bool* success) const -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::Idx<AccCpu>>(1) == (alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());
    }
};

TEST_CASE("kernelNoTemplateCpu", "[kernel]")
{
    alpaka::test::KernelExecutionFixture<AccCpu> fixture(alpaka::Vec<Dim, Idx>::ones());

    KernelNoTemplateCpu kernel;

    REQUIRE(fixture(kernel));
}
#endif

#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
struct KernelNoTemplateGpu
{
    ALPAKA_FN_ACC
    auto operator()(AccGpu const& acc, bool* success) const -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::Idx<AccGpu>>(1) == (alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());
    }
};

TEST_CASE("kernelNoTemplateGpu", "[kernel]")
{
    alpaka::test::KernelExecutionFixture<AccGpu> fixture(alpaka::Vec<Dim, Idx>::ones());

    KernelNoTemplateGpu kernel;

    REQUIRE(fixture(kernel));
}
#endif

#if defined(ALPAKA_ACC_CPU_SERIAL_ENABLED)
struct KernelUnusedTemplateParamCpu
{
    template<typename TNotUsed = void>
    ALPAKA_FN_ACC auto operator()(AccCpu const& acc, bool* success) const -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::Idx<AccCpu>>(1) == (alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());
    }
};

TEST_CASE("kernelUnusedTemplateParamCpu", "[kernel]")
{
    alpaka::test::KernelExecutionFixture<AccCpu> fixture(alpaka::Vec<Dim, Idx>::ones());

    KernelUnusedTemplateParamCpu kernel;

    REQUIRE(fixture(kernel));
}
#endif

#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
struct KernelUnusedTemplateParamGpu
{
    template<typename TNotUsed = void>
    ALPAKA_FN_ACC auto operator()(AccGpu const& acc, bool* success) const -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::Idx<AccGpu>>(1) == (alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());
    }
};

TEST_CASE("kernelUnusedTemplateParamGpu", "[kernel]")
{
    alpaka::test::KernelExecutionFixture<AccGpu> fixture(alpaka::Vec<Dim, Idx>::ones());

    KernelUnusedTemplateParamGpu kernel;

    REQUIRE(fixture(kernel));
}
#endif
