/* Copyright 2022 Sergei Bastrakov, Jan Stephan, Bernhard Manfred Gruber, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/AccCpuOmp2Blocks.hpp>
#include <alpaka/acc/AccCpuOmp2Threads.hpp>
#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/acc/AccCpuTbbBlocks.hpp>
#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/kernel/KernelBundle.hpp>
#include <alpaka/kernel/KernelFunctionAttributes.hpp>
#include <alpaka/math/MathStdLib.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

struct TestKernelWithManyRegisters
{
    template<typename TAcc>
    [[maybe_unused]] ALPAKA_FN_ACC auto operator()(TAcc const& acc, std::size_t val) const -> void
    {
        double var0 = 1.0;
        double var1 = 2.0;
        double var2 = 3.0;

        // Define many variables and use some calculations in order to prevent compiler optimization and make the
        // kernel use many registers (around 80 on sm_52). Using many registers per SM decreases the max number of
        // threads per block while this kernel is being run.

        // TODO: Use function templates to parametrize and shorten the code!
        double var3 = var2 + alpaka::math::fmod(acc, var2, 5);
        double var4 = var3 + alpaka::math::fmod(acc, var3, 5);
        double var5 = var4 + alpaka::math::fmod(acc, var4, 5);
        double var6 = var5 + alpaka::math::fmod(acc, var5, 5);
        double var7 = var6 + alpaka::math::fmod(acc, var6, 5);
        double var8 = var7 + alpaka::math::fmod(acc, var7, 5);
        double var9 = var8 + alpaka::math::fmod(acc, var8, 5);
        double var10 = var9 + alpaka::math::fmod(acc, var9, 5);
        double var11 = var10 + alpaka::math::fmod(acc, var10, 5);
        double var12 = var11 + alpaka::math::fmod(acc, var11, 5);
        double var13 = var12 + alpaka::math::fmod(acc, var12, 5);
        double var14 = var13 + alpaka::math::fmod(acc, var13, 5);
        double var15 = var14 + alpaka::math::fmod(acc, var14, 5);
        double var16 = var15 + alpaka::math::fmod(acc, var15, 5);
        double var17 = var16 + alpaka::math::fmod(acc, var16, 5);
        double var18 = var17 + alpaka::math::fmod(acc, var17, 5);
        double var19 = var18 + alpaka::math::fmod(acc, var18, 5);
        double var20 = var19 + alpaka::math::fmod(acc, var19, 5);
        double var21 = var20 + alpaka::math::fmod(acc, var20, 5);
        double var22 = var21 + alpaka::math::fmod(acc, var21, 5);
        double var23 = var22 + alpaka::math::fmod(acc, var22, 5);
        double var24 = var23 + alpaka::math::fmod(acc, var23, 5);
        double var25 = var24 + alpaka::math::fmod(acc, var24, 5);
        double var26 = var25 + alpaka::math::fmod(acc, var25, 5);
        double var27 = var26 + alpaka::math::fmod(acc, var26, 5);
        double var28 = var27 + alpaka::math::fmod(acc, var27, 5);
        double var29 = var28 + alpaka::math::fmod(acc, var28, 5);
        double var30 = var29 + alpaka::math::fmod(acc, var29, 5);
        double var31 = var30 + alpaka::math::fmod(acc, var30, 5);
        double var32 = var31 + alpaka::math::fmod(acc, var31, 5);
        double var33 = var32 + alpaka::math::fmod(acc, var32, 5);
        double var34 = var33 + alpaka::math::fmod(acc, var33, 5);
        double var35 = var34 + alpaka::math::fmod(acc, var34, 5);

        double sum = var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12
                     + var13 + var14 + var15 + var16 + var17 + var18 + var19 + var20 + var21 + var22 + var23 + var24
                     + var25 + var26 + var27 + var28 + var29 + var30 + var31 + var32 + var33 + var34 + var35;
        printf("The sum is %5.2f, the argument is %lu ", sum, val);
    }
};

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("getValidWorkDivForKernel.1D", "[workDivKernel]", TestAccs)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);

    TestKernelWithManyRegisters kernel;
    auto const bundeledKernel = alpaka::KernelBundle(kernel, 200ul);

    // Get hard limits for test
    auto const props = alpaka::getAccDevProps<Acc, decltype(dev)>(dev);
    Idx const threadsPerGridTestValue = props.m_blockThreadCountMax * props.m_gridBlockCountMax;

    // Test getValidWorkDivForKernel for threadsPerGridTestValue threads per grid
    auto const workDiv
        = alpaka::getValidWorkDivForKernel<Acc>(dev, bundeledKernel, Vec{threadsPerGridTestValue}, Vec{1});
    // Test validity
    auto const isValid = alpaka::isValidWorkDivKernel<Acc>(dev, bundeledKernel, workDiv);
    CHECK(isValid == true);

    if constexpr(alpaka::accMatchesTags<Acc, alpaka::TagGpuCudaRt>)
    {
        // Get calculated threads per block from the workDiv found by examining kernel function
        auto const threadsPerBlock = workDiv.m_blockThreadExtent.prod();
        // Get hard limits
        auto const threadsPerBlockLimit = props.m_blockThreadCountMax;

        // Depending on the GPU type or the compiler the test below might fail because threadsPerBlock can be equal to
        // threadsPerBlockLimit, which is the max device limit.
        CHECK(threadsPerBlock < static_cast<Idx>(threadsPerBlockLimit));
    }
    else if constexpr(alpaka::accMatchesTags<
                          Acc,
                          alpaka::TagGpuHipRt,
                          alpaka::TagCpuThreads,
                          alpaka::TagCpuOmp2Threads,
                          alpaka::TagFpgaSyclIntel,
                          alpaka::TagGpuSyclIntel,
                          alpaka::TagGenericSycl>)
    {
        // Get calculated threads per block from the workDiv found by examining kernel function
        auto const threadsPerBlock = workDiv.m_blockThreadExtent.prod();
        // Get hard limits
        auto const threadsPerBlockLimit = props.m_blockThreadCountMax;

        CHECK(threadsPerBlock <= static_cast<Idx>(threadsPerBlockLimit));
    }
    else if constexpr(alpaka::accMatchesTags<
                          Acc,
                          alpaka::TagCpuSerial,
                          alpaka::TagCpuOmp2Blocks,
                          alpaka::TagCpuTbbBlocks,
                          alpaka::TagCpuSycl>)
    {
        // CPU must have only 1 thread per block. In other words, number of blocks is equal to number of threads.
        CHECK(workDiv == WorkDiv{Vec{threadsPerGridTestValue}, Vec{1}, Vec{1}});
        // Test a new 1D workdiv. Threads per block can not be larger than 1 for CPU. Hence 2 is not valid.
        auto const workDiv1DUsingInitList = WorkDiv{Vec{threadsPerGridTestValue / 2}, Vec{2}, Vec{1}};
        auto const isWorkDivValidForCPU
            = alpaka::isValidWorkDivKernel<Acc>(dev, bundeledKernel, workDiv1DUsingInitList);
        CHECK(isWorkDivValidForCPU == false);
        // Check maxDynamicSharedSizeBytes for CPU backends
        auto const funcAttributes = alpaka::getFunctionAttributes<Acc>(dev, bundeledKernel);
        CHECK(
            funcAttributes.maxDynamicSharedSizeBytes == static_cast<int>(alpaka::BlockSharedDynMemberAllocKiB * 1024));
    }
    else
    {
        throw std::invalid_argument("Acc type is not among tested Accs.");
    }
}

using TestAccs2D = alpaka::test::EnabledAccs<alpaka::DimInt<2u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("getValidWorkDivForKernel.2D", "[workDivKernel]", TestAccs2D)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);

    TestKernelWithManyRegisters kernel;
    // A random value
    size_t val(200ul);
    auto const bundeledKernel = alpaka::KernelBundle(kernel, val);

    // Get hard limits for test
    auto const props = alpaka::getAccDevProps<Acc>(dev);
    Idx const threadsPerGridTestValue = props.m_blockThreadCountMax * props.m_gridBlockCountMax;

    // Test getValidWorkDivForKernel function for threadsPerGridTestValue threads per grid.
    auto const workDiv
        = alpaka::getValidWorkDivForKernel<Acc>(dev, bundeledKernel, Vec{8, threadsPerGridTestValue / 8}, Vec{1, 1});

    // Test isValidWorkDivKernel function
    auto const isValid = alpaka::isValidWorkDivKernel<Acc>(dev, bundeledKernel, workDiv);
    CHECK(isValid == true);

    if constexpr(alpaka::accMatchesTags<Acc, alpaka::TagGpuCudaRt>)
    {
        // Expected valid workdiv values for this kernel might change depending on the GPU type and compiler. Therefore
        // generated workdiv is not compared to a specific workdiv in this test.

        // Get calculated threads per block from the workDiv that was found by examining kernel function
        auto const threadsPerBlock = workDiv.m_blockThreadExtent.prod();
        // Get hard limits
        auto const threadsPerBlockLimit = props.m_blockThreadCountMax;

        // Depending on the GPU type or the compiler the test below might fail because threadsPerBlock can be equal to
        // threadsPerBlockLimit, which is the max device limit.
        CHECK(threadsPerBlock < static_cast<Idx>(threadsPerBlockLimit));

        // too many threads per block
        auto const invalidWorkDiv
            = WorkDiv{Vec{8, threadsPerGridTestValue / 8}, Vec{2 * threadsPerBlock, 1}, Vec{1, 1}};
        auto isWorkDivValidForCuda = alpaka::isValidWorkDivKernel<Acc>(dev, bundeledKernel, invalidWorkDiv);
        CHECK(isWorkDivValidForCuda == false);

        auto const validWorkDiv = WorkDiv{Vec{8, threadsPerGridTestValue / 8}, Vec{1, threadsPerBlock}, Vec{1, 1}};
        isWorkDivValidForCuda = alpaka::isValidWorkDivKernel<Acc>(dev, bundeledKernel, validWorkDiv);
        CHECK(isWorkDivValidForCuda == true);
    }
    else if constexpr(alpaka::accMatchesTags<
                          Acc,
                          alpaka::TagGpuHipRt,
                          alpaka::TagCpuThreads,
                          alpaka::TagCpuOmp2Threads,
                          alpaka::TagFpgaSyclIntel,
                          alpaka::TagGpuSyclIntel,
                          alpaka::TagGenericSycl>)
    {
        // Get calculated threads per block from the workDiv that was found by examining the kernel function
        auto const threadsPerBlock = workDiv.m_blockThreadExtent.prod();
        // Get hard limits
        auto const threadsPerBlockLimit = props.m_blockThreadCountMax;
        // Depending on the GPU type or the compiler this test might fail because threadsPerBlock can be less than
        // threadsPerBlockLimit, which is the max device limit.
        if(threadsPerBlockLimit == 1)
            CHECK(threadsPerBlock == static_cast<Idx>(threadsPerBlockLimit));
        else
            CHECK(threadsPerBlock < static_cast<Idx>(threadsPerBlockLimit));

        // too many threads per block
        auto const invalidWorkDiv
            = WorkDiv{Vec{8, threadsPerGridTestValue / 8}, Vec{20 * threadsPerBlock, 1}, Vec{1, 1}};
        auto isWorkDivValidForHip = alpaka::isValidWorkDivKernel<Acc>(dev, bundeledKernel, invalidWorkDiv);
        CHECK(isWorkDivValidForHip == false);

        auto const validWorkDiv = WorkDiv{Vec{8, threadsPerGridTestValue / 8}, Vec{1, threadsPerBlock}, Vec{1, 1}};
        isWorkDivValidForHip = alpaka::isValidWorkDivKernel<Acc>(dev, bundeledKernel, validWorkDiv);
        CHECK(isWorkDivValidForHip == true);
    }
    else if constexpr(alpaka::accMatchesTags<
                          Acc,
                          alpaka::TagCpuSerial,
                          alpaka::TagCpuOmp2Blocks,
                          alpaka::TagCpuTbbBlocks,
                          alpaka::TagCpuSycl>)
    {
        // CPU must have only 1 thread per block. In other words, number of blocks is equal to number of threads.
        CHECK(workDiv == WorkDiv{Vec{8, threadsPerGridTestValue / 8}, Vec{1, 1}, Vec{1, 1}});
        // Test a new 2D workdiv. Threads per block can not be larger than 1 for CPU. Hence 2x1 threads is not valid.
        auto const invalidWorkDiv2D = WorkDiv{Vec{1, 2048}, Vec{1, 2}, Vec{1, 1}};
        auto const isWorkDivValidForCpu = alpaka::isValidWorkDivKernel<Acc>(dev, bundeledKernel, invalidWorkDiv2D);
        CHECK(isWorkDivValidForCpu == false);

        // Check maxDynamicSharedSizeBytes for CPU backends
        CHECK(
            alpaka::getFunctionAttributes<Acc>(dev, bundeledKernel).maxDynamicSharedSizeBytes
            == static_cast<int>(alpaka::BlockSharedDynMemberAllocKiB * 1024));
    }
    else
    {
        throw std::invalid_argument("Acc type is not among tested Accs.");
    }
}
