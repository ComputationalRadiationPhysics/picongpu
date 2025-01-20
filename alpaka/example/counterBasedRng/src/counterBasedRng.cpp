/* Copyright 2023 Jeffrey Kelling, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>
#include <alpaka/rand/RandPhiloxStateless.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <typeinfo>

//! A kernel that fills an array with pseudo-random numbers.
class CounterBasedRngKernel
{
public:
    template<class TAcc>
    using Vec = alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>>;
    using Gen = typename alpaka::rand::PhiloxStateless4x32x10Vector;
    using Key = typename Gen::Key;
    using Counter = typename Gen::Counter;

    template<typename TAcc, typename TElem>
    using Mdspan = alpaka::experimental::MdSpan<TElem, alpaka::Idx<TAcc>, alpaka::Dim<TAcc>>;

private:
    template<unsigned int I>
    struct ElemLoop
    {
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc, typename TElem>
        static ALPAKA_FN_ACC auto elemLoop(
            TAcc const& acc,
            Mdspan<TAcc, TElem> dst,
            Key const& key,
            Vec<TAcc> const& threadElemExtent,
            Vec<TAcc>& threadFirstElemIdx) -> void
        {
            auto const threadLastElemIdx = threadFirstElemIdx[I] + threadElemExtent[I];
            auto const threadLastElemIdxClipped
                = (dst.extent(I) > threadLastElemIdx) ? threadLastElemIdx : dst.extent(I);

            constexpr auto Dim = alpaka::Dim<TAcc>::value;

            auto const firstElem = threadFirstElemIdx[I];
            if constexpr(I < Dim - 1)
            {
                for(; threadFirstElemIdx[I] < threadLastElemIdxClipped; ++threadFirstElemIdx[I])
                {
                    ElemLoop<I + 1>::elemLoop(acc, dst, key, threadElemExtent, threadFirstElemIdx);
                }
            }
            else
            {
                Counter c = {0, 0, 0, 0};
                for(unsigned int i = 0; i < Dim; ++i)
                    c[i] = threadFirstElemIdx[i];

                for(; threadFirstElemIdx[Dim - 1] < threadLastElemIdxClipped; ++threadFirstElemIdx[Dim - 1])
                {
                    c[Dim - 1] = threadFirstElemIdx[Dim - 1];
                    auto const random = Gen::generate(c, key);
                    // to make use of the whole random vector we would need to ensure numElement[0] % 4 == 0
                    dst(alpaka::toArray(threadFirstElemIdx)) = TElem(random[0]);
                }
            }
            threadFirstElemIdx[I] = firstElem;
        }
    };

public:
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param dst destimation matrix.
    //! \param extent The matrix dimension in elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TElem>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, Mdspan<TAcc, TElem> dst, Key const& key) const -> void
    {
        constexpr auto Dim = alpaka::Dim<TAcc>::value;
        static_assert(Dim <= 4, "The CounterBasedRngKernel expects at most 4-dimensional indices!");

        Vec<TAcc> const gridThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc));
        Vec<TAcc> const threadElemExtent(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc));
        Vec<TAcc> threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        ElemLoop<0>::elemLoop(acc, dst, key, threadElemExtent, threadFirstElemIdx);
    }
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    // Define the index domain
    using Dim = alpaka::DimInt<3u>;
    using Idx = std::size_t;

    // Define the accelerator
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    using AccHost = alpaka::AccCpuSerial<Dim, Idx>;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    using QueueHost = alpaka::Queue<AccHost, QueueProperty>;

    // Select a device
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // Create a queue on the device
    QueueAcc queueAcc(devAcc);
    QueueHost queueHost(devHost);

    // Define the work division
    alpaka::Vec<Dim, Idx> const extent = {16, 16, 16 * 8};
    alpaka::Vec<Dim, Idx> const elementsPerThread = {1, 1, 1};
    alpaka::Vec<Dim, Idx> const elementsPerThreadHost = {1, 1, 8};


    // Define the buffer element type
    using Data = std::uint32_t;

    // Allocate 3 host memory buffers
    auto bufHost(alpaka::allocBuf<Data, Idx>(devHost, extent));
    auto bufHostDev(alpaka::allocBuf<Data, Idx>(devHost, extent));

    std::random_device rd{};
    CounterBasedRngKernel::Key key = {rd(), rd()};

    // Allocate buffer on the accelerator
    using BufAcc = alpaka::Buf<Acc, Data, Dim, Idx>;
    BufAcc bufAcc(alpaka::allocBuf<Data, Idx>(devAcc, extent));

    CounterBasedRngKernel counterBasedRngKernel;

    // Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::KernelCfg<Acc> kernerlCfgAccDev = {extent, elementsPerThread};
    auto const workDivAcc = alpaka::getValidWorkDiv(
        kernerlCfgAccDev,
        devAcc,
        counterBasedRngKernel,
        alpaka::experimental::getMdSpan(bufAcc),
        key);

    // Create the kernel execution task.
    auto const taskKernelAcc = alpaka::createTaskKernel<Acc>(
        workDivAcc,
        CounterBasedRngKernel(),
        alpaka::experimental::getMdSpan(bufAcc),
        key);

    alpaka::KernelCfg<AccHost> kernerlCfgAccHost = {extent, elementsPerThreadHost};
    auto const workDivHost = alpaka::getValidWorkDiv(
        kernerlCfgAccHost,
        devHost,
        counterBasedRngKernel,
        alpaka::experimental::getMdSpan(bufHost),
        key);

    auto const taskKernelHost = alpaka::createTaskKernel<AccHost>(
        workDivHost,
        CounterBasedRngKernel(),
        alpaka::experimental::getMdSpan(bufHost),
        key);

    // Enqueue the kernel execution task
    alpaka::enqueue(queueHost, taskKernelHost);
    alpaka::enqueue(queueAcc, taskKernelAcc);

    // Copy the result from the device
    alpaka::memcpy(queueAcc, bufHostDev, bufAcc);
    auto const numElements = extent.prod();

    // wait in case we are using an asynchronous queue to time actual kernel runtime
    alpaka::wait(queueHost);
    alpaka::wait(queueAcc);

    int falseResults = 0;
    int const maxPrintFalseResults = extent[2] * 2;

    auto aHost = alpaka::experimental::getMdSpan(bufHost);
    auto aAcc = alpaka::experimental::getMdSpan(bufHostDev);
    for(Idx z = 0; z < aHost.extent(0); ++z)
        for(Idx y = 0; y < aHost.extent(1); ++y)
            for(Idx x = 0; x < aHost.extent(2); ++x)
            {
                Data const& valHost(aHost(z, y, x));
                Data const& valAcc(aAcc(z, y, x));
                if(valHost != valAcc)
                {
                    if(falseResults < maxPrintFalseResults)
                        std::cerr << "host[" << z << ", " << y << ", " << x << "] = " << valHost << " != acc[" << z
                                  << ", " << y << ", " << x << "] = " << valAcc << std::endl;
                    ++falseResults;
                }
            }

    if(falseResults == 0)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Found " << falseResults << " false results, printed no more than " << maxPrintFalseResults
                  << "\n"
                  << "Execution results incorrect!" << std::endl;
        return EXIT_FAILURE;
    }
}

auto main() -> int
{
    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks, TagCpuTbbBlocks,
    //   TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks, TagCpuThreads,
    //   TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
