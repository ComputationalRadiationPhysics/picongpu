/* Copyright 2023 Simeon Ehrig, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <string>


using Dim = alpaka::DimInt<1>;
using Idx = int;
using TestAccs = alpaka::test::EnabledAccs<Dim, Idx>;

using TagList = std::tuple<
    alpaka::TagCpuSerial,
    alpaka::TagCpuThreads,
    alpaka::TagCpuTbbBlocks,
    alpaka::TagCpuOmp2Blocks,
    alpaka::TagCpuOmp2Threads,
    alpaka::TagGpuCudaRt,
    alpaka::TagGpuHipRt,
    alpaka::TagCpuSycl,
    alpaka::TagFpgaSyclIntel,
    alpaka::TagGpuSyclIntel>;

using AccToTagMap = std::tuple<
    std::pair<alpaka::test::detail::AccCpuSerialIfAvailableElseInt<Dim, Idx>, alpaka::TagCpuSerial>,
    std::pair<alpaka::test::detail::AccCpuThreadsIfAvailableElseInt<Dim, Idx>, alpaka::TagCpuThreads>,
    std::pair<alpaka::test::detail::AccCpuTbbIfAvailableElseInt<Dim, Idx>, alpaka::TagCpuTbbBlocks>,
    std::pair<alpaka::test::detail::AccCpuOmp2BlocksIfAvailableElseInt<Dim, Idx>, alpaka::TagCpuOmp2Blocks>,
    std::pair<alpaka::test::detail::AccCpuOmp2ThreadsIfAvailableElseInt<Dim, Idx>, alpaka::TagCpuOmp2Threads>,
    std::pair<alpaka::test::detail::AccGpuCudaRtIfAvailableElseInt<Dim, Idx>, alpaka::TagGpuCudaRt>,
    std::pair<alpaka::test::detail::AccGpuHipRtIfAvailableElseInt<Dim, Idx>, alpaka::TagGpuHipRt>,
    std::pair<alpaka::test::detail::AccCpuSyclIfAvailableElseInt<Dim, Idx>, alpaka::TagCpuSycl>,
    std::pair<alpaka::test::detail::AccFpgaSyclIntelIfAvailableElseInt<Dim, Idx>, alpaka::TagFpgaSyclIntel>,
    std::pair<alpaka::test::detail::AccGpuSyclIntelIfAvailableElseInt<Dim, Idx>, alpaka::TagGpuSyclIntel>>;

using AccTagTestMatrix = alpaka::meta::CartesianProduct<std::tuple, AccToTagMap, TagList>;

TEMPLATE_LIST_TEST_CASE("test all possible acc tag combinations", "[acc][tag]", AccTagTestMatrix)
{
    // type of TestType is std::tuple<std::tuple<TAcc, TTag>, TTag>
    // TAcc is ether a alpaka accelerator or int, if not enabled
    // The Tag in in the inner tuple is the expected Tag of the TAcc
    // The Tag in the outer tuple is the Tag to test
    using TestAccTestTuple = std::tuple_element_t<0, TestType>;
    using TestAcc = std::tuple_element_t<0, TestAccTestTuple>;
    using ExpectedTag = std::tuple_element_t<1, TestAccTestTuple>;
    using TestTag = std::tuple_element_t<1, TestType>;

    // if the Acc is not enabled, the type is int
    if constexpr(!std::is_same_v<TestAcc, int>)
    {
        if constexpr(std::is_same_v<TestTag, ExpectedTag>)
        {
            STATIC_REQUIRE((std::is_same_v<alpaka::AccToTag<TestAcc>, ExpectedTag>) );
            STATIC_REQUIRE((std::is_same_v<alpaka::TagToAcc<TestTag, Dim, Idx>, TestAcc>) );
            STATIC_REQUIRE((alpaka::accMatchesTags<TestAcc, TestTag>) );
        }
        else
        {
            STATIC_REQUIRE(!(std::is_same_v<alpaka::AccToTag<TestAcc>, TestTag>) );
            STATIC_REQUIRE(!(alpaka::accMatchesTags<TestAcc, TestTag>) );
        }
    }
}

// ###################################################
// first version of function specialization
// ###################################################

template<typename Tag>
std::string specialized_function()
{
    return "Generic";
}

template<>
std::string specialized_function<alpaka::TagCpuSerial>()
{
    return "Serial";
}

template<>
std::string specialized_function<alpaka::TagGpuCudaRt>()
{
    return "CUDA";
}

// ###################################################
// second version of function specialization
// ###################################################

template<typename TTag>
std::string specialized_function_2(TTag)
{
    return "Generic";
}

// is required because of -Werror=missing-declarations
std::string specialized_function_2(alpaka::TagCpuSerial);

std::string specialized_function_2(alpaka::TagCpuSerial)
{
    return "Serial";
}

std::string specialized_function_2(alpaka::TagGpuCudaRt);

std::string specialized_function_2(alpaka::TagGpuCudaRt)
{
    return "CUDA";
}

TEMPLATE_LIST_TEST_CASE("specialization of functions with tags", "[acc][tag]", TestAccs)
{
    using TestAcc = TestType;

    std::string expected_result = "Generic";

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    if constexpr(std::is_same_v<alpaka::AccCpuSerial<Dim, Idx>, TestAcc>)
    {
        expected_result = "Serial";
    }
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    if constexpr(std::is_same_v<alpaka::AccGpuCudaRt<Dim, Idx>, TestAcc>)
    {
        expected_result = "CUDA";
    }
#endif

    INFO(
        "Test Acc: " + alpaka::getAccName<TestAcc>() + "\n" + "expect: " + expected_result + "\n"
        + "is_specialized() returns: " + specialized_function<alpaka::AccToTag<TestAcc>>() + "\n"
        + "is_specialized_2() returns: " + specialized_function_2(alpaka::AccToTag<TestAcc>{}));
    REQUIRE((specialized_function<alpaka::AccToTag<TestAcc>>() == expected_result));
    REQUIRE((specialized_function_2(alpaka::AccToTag<TestAcc>{}) == expected_result));
}

struct InitKernel
{
    template<typename TAcc, typename TData>
    ALPAKA_FN_ACC auto operator()(TAcc const&, TData* const data) const noexcept -> void
    {
        data[0] = 42;
    }
};

template<typename TAcc, typename = void>
struct specialized_Kernel
{
    template<typename TData>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC TData operator()(TAcc const& acc, TData input)
    {
        return alpaka::math::min(acc, 0, input);
    }
};

template<typename TAcc>
struct specialized_Kernel<TAcc, std::enable_if_t<alpaka::accMatchesTags<TAcc, alpaka::TagGpuCudaRt>>>
{
    template<typename TData>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC TData operator()(TAcc const& acc, TData input)
    {
        return alpaka::math::min(acc, 1, input);
    }
};

template<typename TAcc>
struct specialized_Kernel<
    TAcc,
    std::enable_if_t<alpaka::accMatchesTags<TAcc, alpaka::TagCpuOmp2Blocks, alpaka::TagCpuOmp2Threads>>>
{
    template<typename TData>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC TData operator()(TAcc const& acc, TData input)
    {
        return alpaka::math::min(acc, 2, input);
    }
};

struct WrapperKernel
{
    template<typename TAcc, typename TData>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TData* const data) const noexcept -> void
    {
        data[0] = specialized_Kernel<TAcc>{}(acc, data[0]);
    }
};

TEMPLATE_LIST_TEST_CASE("kernel specialization with tags", "[acc][tag]", TestAccs)
{
    using TestAcc = TestType;
    using QueueProperty = alpaka::Blocking;
    using Queue = alpaka::Queue<TestAcc, QueueProperty>;
    auto const platformAcc = alpaka::Platform<TestAcc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);
    Queue queue(devAcc);

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    using Vec = alpaka::Vec<alpaka::DimInt<1>, int>;
    Vec extents(Vec::all(1));

    auto memHost = alpaka::allocBuf<int, Idx>(devHost, extents);
    auto memAcc = alpaka::allocBuf<int, Idx>(devAcc, extents);
    int* const memAccPtr = alpaka::getPtrNative(memAcc);

    alpaka::WorkDivMembers<Dim, Idx> workdiv{1, 1, 1};

    alpaka::exec<TestAcc>(queue, workdiv, InitKernel{}, memAccPtr);
    alpaka::exec<TestAcc>(queue, workdiv, WrapperKernel{}, memAccPtr);
    alpaka::wait(queue);

    alpaka::memcpy(queue, memHost, memAcc);
    alpaka::wait(queue);

    // result of the generic implementation
    int expected_result = 0;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    if constexpr(std::is_same_v<alpaka::AccGpuCudaRt<Dim, Idx>, TestAcc>)
    {
        expected_result = 1;
    }
#endif

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
    if constexpr(std::is_same_v<alpaka::AccCpuOmp2Threads<Dim, Idx>, TestAcc>)
    {
        expected_result = 2;
    }
#endif

#if defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
    if constexpr(std::is_same_v<alpaka::AccCpuOmp2Blocks<Dim, Idx>, TestAcc>)
    {
        expected_result = 2;
    }
#endif

    REQUIRE(alpaka::getPtrNative(memHost)[0] == expected_result);
}
