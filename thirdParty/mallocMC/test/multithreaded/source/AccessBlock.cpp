/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2024 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Julian Johannes Lenz

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/


#include "mallocMC/creationPolicies/FlatterScatter/AccessBlock.hpp"

#include "mallocMC/mallocMC_utils.hpp"
#include "mocks.hpp"

#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/acc/AccCpuThreads.hpp>
#include <alpaka/acc/Tag.hpp>
#include <alpaka/acc/TagAccIsEnabled.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/mem/alloc/Traits.hpp>
#include <alpaka/mem/buf/BufCpu.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/platform/PlatformCpu.hpp>
#include <alpaka/platform/Traits.hpp>
#include <alpaka/queue/Properties.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <span>
#include <tuple>
#include <type_traits>

using mallocMC::CreationPolicies::FlatterScatterAlloc::AccessBlock;

using Dim = alpaka::DimInt<1>;
using Idx = std::uint32_t;


constexpr uint32_t pageSize = 1024;
constexpr uint32_t numPages = 4;
// Page table entry size = sizeof(chunkSize) + sizeof(fillingLevel):
constexpr uint32_t pteSize = 4 + 4;
constexpr uint32_t blockSize = numPages * (pageSize + pteSize);

using MyAccessBlock = AccessBlock<HeapConfig<blockSize, pageSize>, AlignmentPolicy>;
using std::span;

// Fill all pages of the given access block with occupied chunks of the given size. This is useful to test the
// behaviour near full filling but also to have a deterministic page and chunk where an allocation must happen
// regardless of the underlying access optimisations etc.

struct FillWith
{
    template<typename TAcc, uint32_t T_blockSize, uint32_t T_pageSize>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        AccessBlock<HeapConfig<T_blockSize, T_pageSize>, AlignmentPolicy>* accessBlock,
        uint32_t const chunkSize,
        void** result,
        uint32_t const size) const -> void
    {
        std::generate(
            result,
            result + size,
            [&acc, accessBlock, chunkSize]()
            {
                void* pointer{nullptr};
                while(pointer == nullptr)
                {
                    pointer = accessBlock->create(acc, chunkSize);
                }
                return pointer;
            });
    }
};

struct ContentGenerator
{
    uint32_t counter{0U};

    ALPAKA_FN_ACC auto operator()() -> uint32_t
    {
        return counter++;
    }
};

ALPAKA_FN_ACC auto forAll(auto const& acc, auto size, auto functor)
{
    auto const idx0 = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
    auto const numElements = alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0];
    for(uint32_t i = 0; i < numElements; ++i)
    {
        auto idx = idx0 + i;
        if(idx < size)
        {
            functor(idx);
        }
    }
}

struct Create
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, auto* accessBlock, span<void*> pointers, auto chunkSize) const
    {
        forAll(acc, pointers.size(), [&](auto idx) { pointers[idx] = accessBlock->create(acc, chunkSize); });
    }

    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, auto* accessBlock, span<void*> pointers, auto* chunkSizes) const
    {
        forAll(acc, pointers.size(), [&](auto idx) { pointers[idx] = accessBlock->create(acc, chunkSizes[idx]); });
    }
};

struct CreateUntilSuccess
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, auto* accessBlock, span<void*> pointers, auto chunkSize) const
    {
        forAll(
            acc,
            pointers.size(),
            [&](auto idx)
            {
                while(pointers[idx] == nullptr)
                {
                    pointers[idx] = accessBlock->create(acc, chunkSize);
                }
            });
    }
};

struct Destroy
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, auto* accessBlock, span<void*> pointers) const
    {
        forAll(acc, pointers.size(), [&](auto idx) { accessBlock->destroy(acc, pointers[idx]); });
    }
};

struct IsValid
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        auto* accessBlock,
        void** pointers,
        bool* results,
        uint32_t const size) const
    {
        std::span<void*> tmpPointers(pointers, size);
        std::span<bool> tmpResults(results, size);
        std::transform(
            std::begin(tmpPointers),
            std::end(tmpPointers),
            std::begin(tmpResults),
            [&acc, accessBlock](auto pointer) { return accessBlock->isValid(acc, pointer); });
    }
};

using Host = alpaka::AccCpuSerial<Dim, Idx>;

template<typename TElem, typename TDevHost, typename TDevAcc>
struct Buffer
{
    TDevAcc m_devAcc;
    TDevHost m_devHost;

    alpaka::Vec<Dim, Idx> m_extents;

    alpaka::Buf<TDevAcc, TElem, Dim, Idx> m_onDevice;
    alpaka::Buf<TDevHost, TElem, Dim, Idx> m_onHost;

    Buffer(TDevHost const& devHost, TDevAcc const& devAcc, auto extents)
        : m_devAcc{devAcc}
        , m_devHost{devHost}
        , m_extents{extents}
        , m_onDevice(alpaka::allocBuf<TElem, Idx>(devAcc, m_extents))
        , m_onHost(alpaka::allocBuf<TElem, Idx>(devHost, m_extents))
    {
    }
};

template<typename TElem, typename TDevHost, typename TDevAcc>
auto makeBuffer(TDevHost const& devHost, TDevAcc const& devAcc, auto extents)
{
    return Buffer<TElem, TDevHost, TDevAcc>{devHost, devAcc, extents};
}

auto createChunkSizes(auto const& devHost, auto const& devAcc, auto& queue)
{
    auto chunkSizes = makeBuffer<uint32_t>(devHost, devAcc, 2U);
    chunkSizes.m_onHost[0] = 32U;
    chunkSizes.m_onHost[1] = 512U;
    alpaka::memcpy(queue, chunkSizes.m_onDevice, chunkSizes.m_onHost);
    return chunkSizes;
}

auto createPointers(auto const& devHost, auto const& devAcc, auto& queue, uint32_t const size)
{
    auto pointers = makeBuffer<void*>(devHost, devAcc, size);
    std::span<void*> tmp(alpaka::getPtrNative(pointers.m_onHost), pointers.m_extents[0]);
    std::fill(std::begin(tmp), std::end(tmp), reinterpret_cast<void*>(1U));
    alpaka::memcpy(queue, pointers.m_onDevice, pointers.m_onHost);
    return pointers;
}

template<typename TAcc>
auto setup()
{
    alpaka::Platform<TAcc> const platformAcc = {};
    alpaka::Platform<alpaka::AccCpuSerial<Dim, Idx>> const platformHost = {};
    alpaka::Dev<alpaka::Platform<TAcc>> const devAcc(alpaka::getDevByIdx(platformAcc, 0));
    alpaka::Dev<alpaka::Platform<Host>> const devHost(alpaka::getDevByIdx(platformHost, 0));
    alpaka::Queue<TAcc, alpaka::NonBlocking> queue{devAcc};
    return std::make_tuple(platformAcc, platformHost, devAcc, devHost, queue);
}

template<typename TAcc>
auto createWorkDiv(auto const& devAcc, auto const numElements, auto... args) -> alpaka::WorkDivMembers<Dim, Idx>
{
    if constexpr(std::is_same_v<alpaka::AccToTag<TAcc>, alpaka::TagCpuSerial>)
    {
        return {{1U}, {1U}, {numElements}};
    }
    else
    {
        alpaka::KernelCfg<TAcc> const kernelCfg
            = {numElements, 1, false, alpaka::GridBlockExtentSubDivRestrictions::Unrestricted};
        return alpaka::getValidWorkDiv<TAcc>(kernelCfg, devAcc, args...);
    }
}

template<typename TAcc>
auto fillWith(auto& queue, auto* accessBlock, auto const& chunkSize, auto& pointers)
{
    alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
    alpaka::exec<TAcc>(
        queue,
        workDivSingleThread,
        FillWith{},
        accessBlock,
        chunkSize,
        alpaka::getPtrNative(pointers.m_onDevice),
        pointers.m_extents[0]);
    alpaka::wait(queue);
    alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
    alpaka::wait(queue);
}

template<typename TAcc>
auto fillAllButOne(auto& queue, auto* accessBlock, auto const& chunkSize, auto& pointers)
{
    fillWith<TAcc>(queue, accessBlock, chunkSize, pointers);
    auto* pointer1 = pointers.m_onHost[0];

    // Destroy exactly one pointer (i.e. the first). This is non-destructive on the actual values in
    // devPointers, so we don't need to wait for the copy before to finish.
    alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
    alpaka::exec<TAcc>(
        queue,
        workDivSingleThread,
        Destroy{},
        accessBlock,
        span<void*>(alpaka::getPtrNative(pointers.m_onDevice), 1U));
    alpaka::wait(queue);
    return pointer1;
}

template<typename TAcc, uint32_t T_blockSize, uint32_t T_pageSize>
auto freeAllButOneOnFirstPage(
    auto& queue,
    AccessBlock<HeapConfig<T_blockSize, T_pageSize>, AlignmentPolicy>* accessBlock,
    auto& pointers)
{
    std::span<void*> tmp(alpaka::getPtrNative(pointers.m_onHost), pointers.m_extents[0]);
    std::sort(std::begin(tmp), std::end(tmp));
    // This points to the first chunk of page 0.
    auto* pointer1 = tmp[0];
    alpaka::wait(queue);
    alpaka::memcpy(queue, pointers.m_onDevice, pointers.m_onHost);
    alpaka::wait(queue);
    auto size
        = pointers.m_extents[0] / AccessBlock<HeapConfig<T_blockSize, T_pageSize>, AlignmentPolicy>::numPages() - 1;
    // Delete all other chunks on page 0.
    customExec<TAcc>(
        queue,
        pointers.m_devAcc,
        size,
        Destroy{},
        accessBlock,
        span<void*>(alpaka::getPtrNative(pointers.m_onDevice) + 1U, size));
    alpaka::wait(queue);
    return pointer1;
}

struct CheckContent
{
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto* content, span<void*> pointers, auto* results, auto chunkSize)
        const
    {
        auto const idx0 = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const numElements = alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0];
        for(uint32_t i = 0; i < numElements; ++i)
        {
            auto idx = idx0 + i;
            if(idx < pointers.size())
            {
                auto* begin = reinterpret_cast<uint32_t*>(pointers[idx]);
                auto* end = begin + chunkSize / sizeof(uint32_t);
                results[idx] = std::all_of(begin, end, [idx, content](auto val) { return val == content[idx]; });
            }
        }
    }
};

template<typename TAcc>
auto checkContent(
    auto& devHost,
    auto& devAcc,
    auto& queue,
    auto& pointers,
    auto& content,
    auto& workDiv,
    auto const chunkSize)
{
    auto results = makeBuffer<bool>(devHost, devAcc, pointers.m_extents[0]);
    alpaka::exec<TAcc>(
        queue,
        workDiv,
        CheckContent{},
        alpaka::getPtrNative(content.m_onDevice),
        span<void*>(alpaka::getPtrNative(pointers.m_onDevice), pointers.m_extents[0]),
        alpaka::getPtrNative(results.m_onDevice),
        chunkSize);
    alpaka::wait(queue);
    alpaka::memcpy(queue, results.m_onHost, results.m_onDevice);
    alpaka::wait(queue);


    std::span<bool> tmpResults(alpaka::getPtrNative(results.m_onHost), results.m_extents[0]);
    auto writtenCorrectly = std::reduce(std::cbegin(tmpResults), std::cend(tmpResults), true, std::multiplies<bool>{});

    return writtenCorrectly;
}

struct GetAvailableSlots
{
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto* accessBlock, auto chunkSize, auto* result) const
    {
        *result = accessBlock->getAvailableSlots(acc, chunkSize);
    };
};

template<typename TAcc>
auto getAvailableSlots(auto* accessBlock, auto& queue, auto const& devHost, auto const& devAcc, auto chunkSize)
{
    alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
    alpaka::wait(queue);
    auto result = makeBuffer<uint32_t>(devHost, devAcc, 1U);
    alpaka::wait(queue);
    alpaka::exec<TAcc>(
        queue,
        workDivSingleThread,
        GetAvailableSlots{},
        accessBlock,
        chunkSize,
        alpaka::getPtrNative(result.m_onDevice));
    alpaka::wait(queue);
    alpaka::memcpy(queue, result.m_onHost, result.m_onDevice);
    alpaka::wait(queue);
    auto tmp = result.m_onHost[0];
    alpaka::wait(queue);
    return tmp;
}

template<uint32_t T_blockSize, uint32_t T_pageSize>
auto pageIndex(AccessBlock<HeapConfig<T_blockSize, T_pageSize>, AlignmentPolicy>* accessBlock, auto* pointer)
{
    // This is a bit dirty: What we should do here is enqueue a kernel that calls accessBlock->pageIndex().
    // But we assume that the access block starts with the first page, so the pointer to the first page equals the
    // pointer to the access block. Not sure if this is reliable if the pointers are device pointers.
    return mallocMC::indexOf(pointer, accessBlock, T_pageSize);
}

struct FillAllUpAndWriteToThem
{
    ALPAKA_FN_ACC auto operator()(
        auto const& acc,
        auto* accessBlock,
        auto* content,
        span<void*> pointers,
        auto chunkSize) const
    {
        auto const idx0 = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const numElements = alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0];
        for(uint32_t i = 0; i < numElements; ++i)
        {
            auto idx = idx0 + i;
            if(idx < pointers.size())
            {
                pointers[idx] = accessBlock->create(acc, chunkSize);
                auto* begin = reinterpret_cast<uint32_t*>(pointers[idx]);
                auto* end = begin + chunkSize / sizeof(uint32_t);
                std::fill(begin, end, content[idx]);
            }
        }
    }
};

struct CreateAndDestroMultipleTimes
{
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto* accessBlock, span<void*> pointers, auto chunkSize) const
    {
        forAll(
            acc,
            pointers.size(),
            [&](auto idx)
            {
                pointers[idx] = nullptr;
                for(uint32_t j = 0; j < idx; ++j)
                {
                    // `.isValid()` is not thread-safe, so we use this direct assessment:
                    while(pointers[idx] == nullptr)
                    {
                        pointers[idx] = accessBlock->create(acc, chunkSize);
                    }
                    accessBlock->destroy(acc, pointers[idx]);
                    pointers[idx] = nullptr;
                }
                while(pointers[idx] == nullptr)
                {
                    pointers[idx] = accessBlock->create(acc, chunkSize);
                }
            });
    }
};

struct OversubscribedCreation
{
    uint32_t oversubscriptionFactor{};
    uint32_t availableSlots{};

    ALPAKA_FN_ACC auto operator()(auto const& acc, auto* accessBlock, span<void*> pointers, auto chunkSize) const
    {
        forAll(
            acc,
            pointers.size(),
            [&](auto idx)
            {
                pointers[idx] = nullptr;
                for(uint32_t j = 0; j < idx + 1; ++j)
                {
                    // `.isValid()` is not thread-safe, so we use this direct assessment:
                    while(pointers[idx] == nullptr)
                    {
                        pointers[idx] = accessBlock->create(acc, chunkSize);

                        // CAUTION: The following lines have cost us more than a working day of debugging!
                        // If the hardware you're running on has a single program counter for the whole warp, the whole
                        // warp can't exit the while loop in case of even a single thread requesting another round.
                        // This implies that if we move the `.destroy()` out of the while loop, all the slots get
                        // filled up but the owning threads run idle instead of freeing them up again because they are
                        // waiting for their last companions to give their okay for exiting the loop. This is, of
                        // course, a hopeless endeavour because all slots are filled (we are vastly oversubscribed in
                        // this scenario). So, this loop deadlocks and no thread ever exits.
                        //
                        // ... at least that's what we believe. If you're reading this comment, we might have been
                        // wrong about this.
                        if(pointers[idx] != nullptr)
                        {
                            accessBlock->destroy(acc, pointers[idx]);
                        }
                    }
                    pointers[idx] = nullptr;
                }

                // We only keep some of the memory. In particular, we keep one chunk less than is available,
                // such that threads looking for memory after we've finished can still find some.
                while(pointers[idx] == nullptr and idx > (oversubscriptionFactor - 1) * availableSlots + 1)
                {
                    pointers[idx] = accessBlock->create(acc, chunkSize);
                }
            });
    }
};

struct CreateAllChunkSizes
{
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto* accessBlock, span<void*> pointers, span<uint32_t> chunkSizes)
        const
    {
        forAll(
            acc,
            pointers.size(),
            [&](auto idx)
            {
                pointers[idx] = accessBlock->create(acc, 1U);

                for(auto chunkSize : chunkSizes)
                {
                    accessBlock->destroy(acc, pointers[idx]);
                    pointers[idx] = nullptr;

                    // `.isValid()` is not thread-safe, so we use this direct assessment:
                    while(pointers[idx] == nullptr)
                    {
                        pointers[idx] = accessBlock->create(acc, chunkSize);
                    }
                }
            });
    }
};

template<typename TAcc>
auto customExec(auto& queue, auto const& devAcc, auto const numElements, auto... args)
{
    auto workDiv = createWorkDiv<TAcc>(devAcc, numElements, args...);
    alpaka::exec<TAcc>(queue, workDiv, args...);
    return workDiv;
}

TEMPLATE_LIST_TEST_CASE("Threaded AccessBlock", "", alpaka::EnabledAccTags)
{
    using Acc = alpaka::TagToAcc<TestType, Dim, Idx>;
    auto [platformAcc, platformHost, devAcc, devHost, queue] = setup<Acc>();
    auto accessBlockBuf = alpaka::allocBuf<MyAccessBlock, Idx>(devAcc, alpaka::Vec<Dim, Idx>{1U});
    alpaka::memset(queue, accessBlockBuf, 0x00);
    alpaka::wait(queue);
    auto* accessBlock = alpaka::getPtrNative(accessBlockBuf);
    auto const chunkSizes = createChunkSizes(devHost, devAcc, queue);
    auto pointers = createPointers(
        devHost,
        devAcc,
        queue,
        getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]));
    alpaka::wait(queue);

    SECTION("creates second memory somewhere else.")
    {
        uint32_t const size = 2U;
        customExec<Acc>(
            queue,
            devAcc,
            size,
            Create{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), size),
            chunkSizes.m_onHost[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        CHECK(pointers.m_onHost[0] != pointers.m_onHost[1]);
    }

    SECTION("creates memory of different chunk size in different pages.")
    {
        customExec<Acc>(
            queue,
            devAcc,
            2U,
            Create{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), 2U),
            alpaka::getPtrNative(chunkSizes.m_onDevice));
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        CHECK(pageIndex(accessBlock, pointers.m_onHost[0]) != pageIndex(accessBlock, pointers.m_onHost[1]));
    }

    SECTION("creates partly for insufficient memory with same chunk size.")
    {
        uint32_t const size = 2U;
        auto* lastFreeChunk = fillAllButOne<Acc>(queue, accessBlock, chunkSizes.m_onHost[0], pointers);

        // Okay, so here we start the actual test. The situation is the following:
        // There is a single chunk available.
        // We try to do two allocations.
        // So, we expect one to succeed and one to fail.
        customExec<Acc>(
            queue,
            devAcc,
            size,
            Create{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), size),
            chunkSizes.m_onHost[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        CHECK(
            ((pointers.m_onHost[0] == lastFreeChunk and pointers.m_onHost[1] == nullptr)
             or (pointers.m_onHost[1] == lastFreeChunk and pointers.m_onHost[0] == nullptr)));
    }

    SECTION("does not race between clean up and create.")
    {
        fillWith<Acc>(queue, accessBlock, chunkSizes.m_onHost[0], pointers);
        auto freePage = pageIndex(accessBlock, freeAllButOneOnFirstPage<Acc>(queue, accessBlock, pointers));

        // Now, pointer1 is the last valid pointer to page 0. Destroying it will clean up the page.
        alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};

        alpaka::exec<Acc>(
            queue,
            workDivSingleThread,
            Destroy{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), pointers.m_extents[0]));

        alpaka::exec<Acc>(
            queue,
            workDivSingleThread,
            CreateUntilSuccess{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), 1U),
            chunkSizes.m_onHost[0]);

        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        CHECK(pageIndex(accessBlock, pointers.m_onHost[0]) == freePage);
    }

    SECTION("destroys two pointers of different size.")
    {
        customExec<Acc>(
            queue,
            devAcc,
            2U,
            Create{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), 2U),
            alpaka::getPtrNative(chunkSizes.m_onDevice));
        alpaka::wait(queue);

        customExec<Acc>(
            queue,
            devAcc,
            2U,
            Destroy{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), 2U));
        alpaka::wait(queue);

        auto result = makeBuffer<bool>(devHost, devAcc, 2U);
        customExec<Acc>(
            queue,
            devAcc,
            1U,
            IsValid{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            alpaka::getPtrNative(result.m_onDevice),
            result.m_extents[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, result.m_onHost, result.m_onDevice);
        alpaka::wait(queue);

        CHECK(not result.m_onHost[0]);
        CHECK(not result.m_onHost[1]);
    }

    SECTION("destroys two pointers of same size.")
    {
        customExec<Acc>(
            queue,
            devAcc,
            2U,
            Create{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), 2U),
            chunkSizes.m_onHost[0]);
        alpaka::wait(queue);

        customExec<Acc>(
            queue,
            devAcc,
            2U,
            Destroy{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), 2U));
        alpaka::wait(queue);

        auto result = makeBuffer<bool>(devHost, devAcc, 2U);
        result.m_onHost[0] = true;
        result.m_onHost[1] = true;
        alpaka::memcpy(queue, result.m_onDevice, result.m_onHost);
        alpaka::wait(queue);
        customExec<Acc>(
            queue,
            devAcc,
            1U,
            IsValid{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            alpaka::getPtrNative(result.m_onDevice),
            result.m_extents[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, result.m_onHost, result.m_onDevice);
        alpaka::wait(queue);

        CHECK(not result.m_onHost[0]);
        CHECK(not result.m_onHost[1]);
    }

    SECTION("fills up all chunks in parallel and writes to them.")
    {
        auto content = makeBuffer<uint32_t>(
            devHost,
            devAcc,
            getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]));
        std::span<uint32_t> tmp(alpaka::getPtrNative(content.m_onHost), content.m_extents[0]);
        std::generate(std::begin(tmp), std::end(tmp), ContentGenerator{});
        alpaka::memcpy(queue, content.m_onDevice, content.m_onHost);
        alpaka::wait(queue);

        auto workDiv = customExec<Acc>(
            queue,
            devAcc,
            pointers.m_extents[0],
            FillAllUpAndWriteToThem{},
            accessBlock,
            alpaka::getPtrNative(content.m_onDevice),
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), pointers.m_extents[0]),
            chunkSizes.m_onHost[0]);

        alpaka::wait(queue);

        auto writtenCorrectly
            = checkContent<Acc>(devHost, devAcc, queue, pointers, content, workDiv, chunkSizes.m_onHost[0]);
        CHECK(writtenCorrectly);
    }

    SECTION("destroys all pointers simultaneously.")
    {
        auto const allSlots = getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]);
        auto const allSlotsOfDifferentSize
            = getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[1]);
        fillWith<Acc>(queue, accessBlock, chunkSizes.m_onHost[0], pointers);

        customExec<Acc>(
            queue,
            devAcc,
            pointers.m_extents[0],
            Destroy{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), pointers.m_extents[0]));
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        auto result = makeBuffer<bool>(devHost, devAcc, pointers.m_extents[0]);
        customExec<Acc>(
            queue,
            devAcc,
            1U,
            IsValid{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            alpaka::getPtrNative(result.m_onDevice),
            result.m_extents[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, result.m_onHost, result.m_onDevice);
        alpaka::wait(queue);

        std::span<bool> tmpResults(alpaka::getPtrNative(result.m_onHost), result.m_extents[0]);
        CHECK(std::none_of(std::cbegin(tmpResults), std::cend(tmpResults), [](auto const val) { return val; }));

        CHECK(getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]) == allSlots);
        CHECK(
            getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[1])
            == allSlotsOfDifferentSize);
    }

    SECTION("creates and destroys multiple times.")
    {
        customExec<Acc>(
            queue,
            devAcc,
            pointers.m_extents[0],
            CreateAndDestroMultipleTimes{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), pointers.m_extents[0]),
            chunkSizes.m_onHost[0]);
        alpaka::wait(queue);
        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        std::span<void*> tmpPointers(alpaka::getPtrNative(pointers.m_onHost), pointers.m_extents[0]);
        std::sort(std::begin(tmpPointers), std::end(tmpPointers));
        CHECK(std::unique(std::begin(tmpPointers), std::end(tmpPointers)) == std::end(tmpPointers));
    }

    SECTION("can handle oversubscription.")
    {
        uint32_t oversubscriptionFactor = 2U;
        auto availableSlots = getAvailableSlots<Acc>(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]);

        // This is oversubscribed but we will only hold keep less than 1/oversubscriptionFactor of the memory in the
        // end.
        auto manyPointers = makeBuffer<void*>(devHost, devAcc, oversubscriptionFactor * availableSlots);
        customExec<Acc>(
            queue,
            devAcc,
            manyPointers.m_extents[0],
            OversubscribedCreation{oversubscriptionFactor, availableSlots},
            accessBlock,
            span<void*>(alpaka::getPtrNative(manyPointers.m_onDevice), manyPointers.m_extents[0]),
            chunkSizes.m_onHost[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, manyPointers.m_onHost, manyPointers.m_onDevice);
        alpaka::wait(queue);

        // We only let the last (availableSlots-1) keep their memory. So, the rest at the beginning should have a
        // nullptr.
        std::span<void*> tmpManyPointers(alpaka::getPtrNative(manyPointers.m_onHost), manyPointers.m_extents[0]);
        auto beginNonNull = std::begin(tmpManyPointers) + (oversubscriptionFactor - 1) * availableSlots + 1;

        CHECK(std::all_of(
            std::begin(tmpManyPointers),
            beginNonNull,
            [](auto const pointer) { return pointer == nullptr; }));

        std::sort(beginNonNull, std::end(tmpManyPointers));
        CHECK(std::unique(beginNonNull, std::end(tmpManyPointers)) == std::end(tmpManyPointers));
    }

    SECTION("can handle many different chunk sizes.")
    {
        auto chunkSizes = makeBuffer<uint32_t>(devHost, devAcc, pageSize);
        std::span<uint32_t> chunkSizesSpan(alpaka::getPtrNative(chunkSizes.m_onHost), chunkSizes.m_extents[0]);
        std::iota(std::begin(chunkSizesSpan), std::end(chunkSizesSpan), 1U);
        alpaka::memcpy(queue, chunkSizes.m_onDevice, chunkSizes.m_onHost);
        alpaka::wait(queue);

        customExec<Acc>(
            queue,
            devAcc,
            MyAccessBlock::numPages(),
            CreateAllChunkSizes{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), MyAccessBlock::numPages()),
            std::span<uint32_t>(alpaka::getPtrNative(chunkSizes.m_onDevice), chunkSizes.m_extents[0]));

        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        std::span<void*> tmpPointers(alpaka::getPtrNative(pointers.m_onHost), MyAccessBlock::numPages());
        std::sort(std::begin(tmpPointers), std::end(tmpPointers));
        CHECK(std::unique(std::begin(tmpPointers), std::end(tmpPointers)) == std::end(tmpPointers));
    }

    SECTION("creates second memory somewhere in multi-page mode.")
    {
        uint32_t const size = 2U;
        customExec<Acc>(
            queue,
            devAcc,
            size,
            Create{},
            accessBlock,
            span<void*>(alpaka::getPtrNative(pointers.m_onDevice), size),
            pageSize);
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        CHECK(pointers.m_onHost[0] != pointers.m_onHost[1]);
    }

    alpaka::wait(queue);
}
