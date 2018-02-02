/**
 * \file
 * Copyright 2015 Benjamin Worpitz
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

#include <alpaka/alpaka.hpp>

#include <alpaka/test/KernelExecutionFixture.hpp>

#include <boost/predef.h>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

BOOST_AUTO_TEST_SUITE(kernel)

//#############################################################################
//! It is not possible to use a alpaka kernel function object without a templated operator() when the CUDA accelerator is hard-coded.
//!
//! However, compiling such kernels with a CPU device works fine.
//!
//! When the CUDA accelerator is used, the following error is triggered:
//! /alpaka/include/alpaka/workdiv/Traits.hpp(...): error: calling a __device__ function("getWorkDiv") from a __host__ __device__ function("getWorkDiv") is not allowed
//! The kernel function objects function call operator is attributed with ALPAKA_FN_ACC which is identical to __host__ __device__.
//! The 'alpaka::workdiv::getWorkDiv<...>(acc)' function that is called has the ALPAKA_FN_HOST_ACC attribute (also equal to __host__ __device__).
//! The underlying trait calls the CUDA specialized method which has the ALPAKA_FN_ACC_CUDA_ONLY attribute (equal to __device__).
//! Because this call chain does not contain any templates and therefore no calls depending on input types,
//! everything can be resolved at the first time the template is parsed which results in the given error.
//!
//! Currently, the only possible way to solve this is to make the function call operator a template nonetheless by providing an unused template parameter.

using Dim = alpaka::dim::DimInt<2u>;
using Size = std::uint32_t;
#if !defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE)
using AccCpu = alpaka::acc::AccCpuSerial<Dim, Size>;
#endif
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
using AccGpu = alpaka::acc::AccGpuCudaRt<Dim, Size>;
#endif

#if !defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE)
//#############################################################################
struct KernelNoTemplateCpu
{
    //-----------------------------------------------------------------------------
    ALPAKA_FN_ACC
    auto operator()(
        AccCpu const & acc) const
    -> void
    {
        // Do something useless on the accelerator.
        alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
    }
};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(kernelNoTemplateCpu)
{
    alpaka::test::KernelExecutionFixture<AccCpu> fixture(
        alpaka::vec::Vec<Dim, Size>::ones());

    KernelNoTemplateCpu kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
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
        AccGpu const & acc) const
    -> void
    {
        // Do something useless on the accelerator.
        alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
    }
};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(kernelNoTemplateGpu)
{
    alpaka::test::KernelExecutionFixture<AccGpu> fixture(
        alpaka::vec::Vec<Dim, Size>::ones());

    KernelNoTemplateGpu kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}
#endif*/

#if !defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE)
//#############################################################################
struct KernelWithoutTemplateParamCpu
{
    //-----------------------------------------------------------------------------
    template<
        typename TNotUsed = void>
    ALPAKA_FN_ACC
    auto operator()(
        AccCpu const & acc) const
    -> void
    {
        // Do something useless on the accelerator.
        alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
    }
};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(kernelWithoutTemplateParamCpu)
{
    alpaka::test::KernelExecutionFixture<AccCpu> fixture(
        alpaka::vec::Vec<Dim, Size>::ones());

    KernelWithoutTemplateParamCpu kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
//#############################################################################
struct KernelWithoutTemplateParamGpu
{
    //-----------------------------------------------------------------------------
    template<
        typename TNotUsed = void>
    ALPAKA_FN_ACC
    auto operator()(
        AccGpu const & acc) const
    -> void
    {
        // Do something useless on the accelerator.
        alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
    }
};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(kernelWithoutTemplateParamGpu)
{
    alpaka::test::KernelExecutionFixture<AccGpu> fixture(
        alpaka::vec::Vec<Dim, Size>::ones());

    KernelWithoutTemplateParamGpu kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}
#endif

BOOST_AUTO_TEST_SUITE_END()
