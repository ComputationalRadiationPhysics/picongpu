/* Copyright 2026 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/warp/Traits.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)

#    include <sycl/sycl.hpp>

template<std::uint32_t TWarpSize>
struct SyclCpuWarpLayoutTestKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        ALPAKA_CHECK(*success, alpaka::warp::getSize(acc) == static_cast<std::int32_t>(TWarpSize));
    }
};

template<std::uint32_t TWarpSize, typename TAcc>
struct alpaka::trait::WarpSize<SyclCpuWarpLayoutTestKernel<TWarpSize>, TAcc>
    : std::integral_constant<std::uint32_t, TWarpSize>
{
};

// On the SYCL CPU device sub-groups are formed only along the innermost dimension of the work-group, so a kernel that
// requires a warp size must be launched with an innermost block extent that is a multiple of the warp size.
template<std::uint32_t TWarpSize>
void testSyclCpuWarpLayout()
{
    using Dim = alpaka::DimInt<2u>;
    using Idx = std::uint32_t;
    using Acc = alpaka::AccCpuSycl<Dim, Idx>;
    using Vec = alpaka::Vec<Dim, Idx>;
    using ExecutionFixture = alpaka::test::KernelExecutionFixture<Acc>;

    // the warp is laid out along the innermost dimension: valid
    auto good = ExecutionFixture{typename ExecutionFixture::WorkDiv{Vec{2, 2}, Vec{1, TWarpSize}, Vec{1, 1}}};
    CHECK(good(SyclCpuWarpLayoutTestKernel<TWarpSize>{}));

    // the warp is laid out along the outer dimension: rejected when the kernel is launched
    auto bad = ExecutionFixture{typename ExecutionFixture::WorkDiv{Vec{2, 2}, Vec{TWarpSize, 1}, Vec{1, 1}}};
    CHECK_THROWS_AS(bad(SyclCpuWarpLayoutTestKernel<TWarpSize>{}), std::runtime_error);
}

TEST_CASE("syclCpuWarpLayout", "[warp]")
{
    using Acc = alpaka::AccCpuSycl<alpaka::DimInt<2u>, std::uint32_t>;
    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    for(auto const warpSize : alpaka::getWarpSizes(dev))
    {
        if(warpSize == 4)
            testSyclCpuWarpLayout<4>();
        else if(warpSize == 8)
            testSyclCpuWarpLayout<8>();
        else if(warpSize == 16)
            testSyclCpuWarpLayout<16>();
        else if(warpSize == 32)
            testSyclCpuWarpLayout<32>();
    }
}

#endif
