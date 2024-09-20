/* Copyright 2022  René Widera, Mehmet Yusufoglu, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/RuntimeMacros.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

class KernelWithThrow
{
public:
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
        using Idx = alpaka::Idx<TAcc>;
        using Dim = alpaka::Dim<TAcc>;
        using Vec = alpaka::Vec<Dim, Idx>;
        Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        if(globalThreadIdx == Vec::zeros())
        {
            // Throw abort or std::runtime_error depending on acc type
            ALPAKA_THROW_ACC("Exception thrown by the kernel.");
        }
        alpaka::syncBlockThreads(acc);
    }
};

template<typename T, typename Acc>
void checkThrow(std::string const& expectedErrStr)
{
    if constexpr(alpaka::accMatchesTags<Acc, T>)
    {
        using Idx = alpaka::Idx<Acc>;
        using Dim = alpaka::Dim<Acc>;
        using Vec = alpaka::Vec<Dim, Idx>;
        using Queue = alpaka::Queue<Acc, alpaka::Blocking>;

        auto const platformAcc = alpaka::Platform<Acc>{};
        auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

        Queue queue(devAcc);
        auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{Vec{8}, Vec{1}, Vec{1}};

        try
        {
            alpaka::exec<Acc>(queue, workDiv, KernelWithThrow{});
            // Cuda can catch exceptions which were thrown at kernel during the wait(); therefore wait is added.
            alpaka::wait(queue);
        }
        catch(std::runtime_error& e)
        {
            std::string const errorStr{e.what()};
            printf("The error str catched: %s \n", errorStr.c_str());
            printf("The expected str in error str: %s \n", expectedErrStr.c_str());

            auto const found = errorStr.find(expectedErrStr);
            CHECK(found != std::string::npos);
        }
        catch(std::exception& e)
        {
            FAIL(std::string("Wrong exception type thrown in kernel:") + e.what());
        }
    }
}

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("ThrowForCpuThreadAndSerial", "[runtime]", TestAccs)
{
    using Acc = TestType;
    // Test runtime-error exceptions.
    checkThrow<alpaka::TagCpuThreads, Acc>("Exception thrown by the kernel");
    checkThrow<alpaka::TagCpuSerial, Acc>("Exception thrown by the kernel");
}

TEMPLATE_LIST_TEST_CASE("ThrowForGpuBackend", "[runtime]", TestAccs)
{
    using Acc = TestType;
    // Test runtime-error exceptions.
    checkThrow<alpaka::TagGpuCudaRt, Acc>("cudaErrorLaunchFailure");
}
