/* Copyright 2022 Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <alpaka/mem/view/Accessor.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/mem/view/ViewStdArray.hpp>
#include <alpaka/mem/view/ViewStdVector.hpp>
#include <alpaka/mem/view/ViewSubView.hpp>

#include <catch2/catch_test_macros.hpp>

namespace alpakaex = alpaka::experimental;

TEST_CASE("IsView", "[accessor]")
{
    using alpakaex::trait::internal::IsView;

    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    using Dev = alpaka::Dev<Acc>;

    // buffer
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto buffer = alpaka::allocBuf<int, Size>(devAcc, Size{1});
    STATIC_REQUIRE(IsView<decltype(buffer)>::value);

    // views
    STATIC_REQUIRE(IsView<alpaka::ViewPlainPtr<Dev, int, Dim, Size>>::value);
    STATIC_REQUIRE(IsView<std::array<int, 42>>::value);
    STATIC_REQUIRE(IsView<std::vector<int>>::value);
    STATIC_REQUIRE(IsView<alpaka::ViewSubView<Dev, int, Dim, Size>>::value);

    // accessor
    auto accessor = alpakaex::access(buffer);
    STATIC_REQUIRE(!IsView<decltype(accessor)>::value);
}

namespace
{
    constexpr auto N = 1024;

    struct WriteKernelTemplate
    {
        template<typename TAcc, typename TAccessor>
        ALPAKA_FN_ACC void operator()(TAcc const&, TAccessor data) const
        {
            data[1] = 1.0f;
            data(2) = 2.0f;
            data[alpaka::Vec<alpaka::DimInt<1>, alpaka::Idx<TAcc>>{alpaka::Idx<TAcc>{3}}] = 3.0f;
        }
    };

    struct WriteKernelExplicit
    {
        template<typename TAcc, typename TMemoryHandle, typename TIdx>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            alpakaex::Accessor<TMemoryHandle, float, TIdx, 1, alpakaex::WriteAccess> const data) const
        {
            data[1] = 1.0f;
            data(2) = 2.0f;
            data[alpaka::Vec<alpaka::DimInt<1>, TIdx>{TIdx{3}}] = 3.0f;
        }
    };

    struct ReadKernelTemplate
    {
        template<typename TAcc, typename TAccessor>
        ALPAKA_FN_ACC void operator()(TAcc const&, TAccessor data) const
        {
            float const v1 = data[1];
            float const v2 = data(2);
            float const v3 = data[alpaka::Vec<alpaka::DimInt<1>, alpaka::Idx<TAcc>>{alpaka::Idx<TAcc>{3}}];
            (void) v1;
            (void) v2;
            (void) v3;
        }
    };

    struct ReadKernelExplicit
    {
        template<typename TAcc, typename TMemoryHandle, typename TIdx>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            alpakaex::Accessor<TMemoryHandle, float, TIdx, 1, alpakaex::ReadAccess> const data) const
        {
            float const v1 = data[1];
            float const v2 = data(2);
            float const v3 = data[alpaka::Vec<alpaka::DimInt<1>, TIdx>{TIdx{3}}];
            (void) v1;
            (void) v2;
            (void) v3;
        }
    };

    struct ReadWriteKernelExplicit
    {
        template<typename TAcc, typename TMemoryHandle, typename TIdx>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            alpakaex::Accessor<TMemoryHandle, float, TIdx, 1, alpakaex::ReadWriteAccess> const data) const
        {
            float const v1 = data[1];
            float const v2 = data(2);
            float const v3 = data[alpaka::Vec<alpaka::DimInt<1>, TIdx>{TIdx{3}}];
            (void) v1;
            (void) v2;
            (void) v3;

            data[1] = 1.0f;
            data(2) = 2.0f;
            data[alpaka::Vec<alpaka::DimInt<1>, TIdx>{TIdx{3}}] = 3.0f;
        }
    };
} // namespace

TEST_CASE("readWrite", "[accessor]")
{
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    using DevAcc = alpaka::Dev<Acc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;

    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto queue = Queue{devAcc};
    auto buffer = alpaka::allocBuf<float, Size>(devAcc, Size{N});
    auto const workdiv = alpaka::WorkDivMembers<Dim, Size>{
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}}};

    alpaka::exec<Acc>(queue, workdiv, WriteKernelTemplate{}, alpakaex::writeAccess(buffer));
    alpaka::exec<Acc>(queue, workdiv, WriteKernelExplicit{}, alpakaex::writeAccess(buffer));
    alpaka::exec<Acc>(queue, workdiv, ReadKernelTemplate{}, alpakaex::readAccess(buffer));
    alpaka::exec<Acc>(queue, workdiv, ReadKernelExplicit{}, alpakaex::readAccess(buffer));
    alpaka::exec<Acc>(queue, workdiv, ReadWriteKernelExplicit{}, alpakaex::access(buffer));
}

namespace
{
    template<typename TProjection, typename TMemoryHandle, typename TElem, typename TBufferIdx, std::size_t TDim>
    struct AccessorWithProjection;

    template<typename TProjection, typename TMemoryHandle, typename TElem, typename TBufferIdx>
    struct AccessorWithProjection<TProjection, TMemoryHandle, TElem, TBufferIdx, 1>
    {
        ALPAKA_FN_ACC auto operator[](alpaka::Vec<alpaka::DimInt<1>, TBufferIdx> i) const -> TElem
        {
            return TProjection{}(accessor[i]);
        }

        ALPAKA_FN_ACC auto operator[](TBufferIdx i) const -> TElem
        {
            return TProjection{}(accessor[i]);
        }

        ALPAKA_FN_ACC auto operator()(TBufferIdx i) const -> TElem
        {
            return TProjection{}(accessor(i));
        }

        alpakaex::Accessor<TMemoryHandle, TElem, TBufferIdx, 1, alpakaex::ReadAccess> accessor;
    };

    template<typename TProjection, typename TMemoryHandle, typename TElem, typename TBufferIdx>
    struct AccessorWithProjection<TProjection, TMemoryHandle, TElem, TBufferIdx, 2>
    {
        ALPAKA_FN_ACC auto operator[](alpaka::Vec<alpaka::DimInt<2>, TBufferIdx> i) const -> TElem
        {
            return TProjection{}(accessor[i]);
        }

        ALPAKA_FN_ACC auto operator()(TBufferIdx y, TBufferIdx x) const -> TElem
        {
            return TProjection{}(accessor(y, x));
        }

        alpakaex::Accessor<TMemoryHandle, TElem, TBufferIdx, 2, alpakaex::ReadAccess> accessor;
    };

    template<typename TProjection, typename TMemoryHandle, typename TElem, typename TBufferIdx>
    struct AccessorWithProjection<TProjection, TMemoryHandle, TElem, TBufferIdx, 3>
    {
        ALPAKA_FN_ACC auto operator[](alpaka::Vec<alpaka::DimInt<3>, TBufferIdx> i) const -> TElem
        {
            return TProjection{}(accessor[i]);
        }

        ALPAKA_FN_ACC auto operator()(TBufferIdx z, TBufferIdx y, TBufferIdx x) const -> TElem
        {
            return TProjection{}(accessor(z, y, x));
        }

        alpakaex::Accessor<TMemoryHandle, TElem, TBufferIdx, 3, alpakaex::ReadAccess> accessor;
    };

    struct DoubleValue
    {
        ALPAKA_FN_ACC auto operator()(int i) const
        {
            return i * 2;
        }
    };

    struct CopyKernel
    {
        template<typename TAcc, typename TMemoryHandle, typename TIdx>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            alpakaex::Accessor<TMemoryHandle, int, TIdx, 1, alpakaex::ReadAccess> const src,
            alpakaex::Accessor<TMemoryHandle, int, TIdx, 1, alpakaex::WriteAccess> const dst) const
        {
            auto const projSrc = AccessorWithProjection<DoubleValue, TMemoryHandle, int, TIdx, 1>{src};
            dst[0] = projSrc[0];
        }
    };
} // namespace

TEST_CASE("projection", "[accessor]")
{
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    using DevAcc = alpaka::Dev<Acc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;

    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto queue = Queue{devAcc};

    auto srcBuffer = alpaka::allocBuf<int, Size>(devAcc, Size{1});
    auto dstBuffer = alpaka::allocBuf<int, Size>(devAcc, Size{1});

    std::array<int, 1> host{{42}};
    alpaka::memcpy(queue, srcBuffer, host);

    auto const workdiv = alpaka::WorkDivMembers<Dim, Size>{
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}}};
    alpaka::exec<Acc>(queue, workdiv, CopyKernel{}, alpakaex::readAccess(srcBuffer), alpakaex::writeAccess(dstBuffer));

    alpaka::memcpy(queue, host, dstBuffer);

    REQUIRE(host[0] == 84);
}

TEST_CASE("constraining", "[accessor]")
{
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;

    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto buffer = alpaka::allocBuf<int, Size>(devAcc, Size{1});
    using MemoryHandle = alpakaex::MemoryHandle<decltype(alpakaex::access(buffer))>;

    alpakaex::Accessor<
        MemoryHandle,
        int,
        Size,
        1,
        std::tuple<alpakaex::ReadAccess, alpakaex::WriteAccess, alpakaex::ReadWriteAccess>>
        acc = alpakaex::accessWith<alpakaex::ReadAccess, alpakaex::WriteAccess, alpakaex::ReadWriteAccess>(buffer);

    // constraining from multi-tag to single-tag
    alpakaex::Accessor<MemoryHandle, int, Size, 1, alpakaex::ReadAccess> readAcc = alpakaex::readAccess(acc);
    alpakaex::Accessor<MemoryHandle, int, Size, 1, alpakaex::WriteAccess> writeAcc = alpakaex::writeAccess(acc);
    alpakaex::Accessor<MemoryHandle, int, Size, 1, alpakaex::ReadWriteAccess> readWriteAcc = alpakaex::access(acc);
    (void) readAcc;
    (void) writeAcc;
    (void) readWriteAcc;

    // constraining from single-tag to single-tag
    alpakaex::Accessor<MemoryHandle, int, Size, 1, alpakaex::ReadAccess> readAcc2 = alpakaex::readAccess(readAcc);
    alpakaex::Accessor<MemoryHandle, int, Size, 1, alpakaex::WriteAccess> writeAcc2 = alpakaex::writeAccess(writeAcc);
    alpakaex::Accessor<MemoryHandle, int, Size, 1, alpakaex::ReadWriteAccess> readWriteAcc2
        = alpakaex::access(readWriteAcc);
    (void) readAcc2;
    (void) writeAcc2;
    (void) readWriteAcc2;
}

namespace
{
    struct BufferAccessorKernelRead
    {
        template<typename TAcc, typename TMemoryHandle, typename TIdx>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            alpakaex::Accessor<TMemoryHandle, int, TIdx, 1, alpakaex::ReadAccess> const r1,
            alpakaex::BufferAccessor<TAcc, int, 1, alpakaex::ReadAccess> const r2,
            alpakaex::BufferAccessor<TAcc, int, 1, alpakaex::ReadAccess, TIdx> const r3) const noexcept
        {
            static_assert(std::is_same_v<decltype(r1), decltype(r2)>);
            static_assert(std::is_same_v<decltype(r2), decltype(r3)>);
        }
    };

    struct BufferAccessorKernelWrite
    {
        template<typename TAcc, typename TMemoryHandle, typename TIdx>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            alpakaex::Accessor<TMemoryHandle, int, TIdx, 1, alpakaex::WriteAccess> const w1,
            alpakaex::BufferAccessor<TAcc, int, 1, alpakaex::WriteAccess> const w2,
            alpakaex::BufferAccessor<TAcc, int, 1, alpakaex::WriteAccess, TIdx> const w3) const noexcept
        {
            static_assert(std::is_same_v<decltype(w1), decltype(w2)>);
            static_assert(std::is_same_v<decltype(w2), decltype(w3)>);
        }
    };
    struct BufferAccessorKernelReadWrite
    {
        template<typename TAcc, typename TMemoryHandle, typename TIdx>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            alpakaex::Accessor<TMemoryHandle, int, TIdx, 1, alpakaex::ReadWriteAccess> const rw1,
            alpakaex::BufferAccessor<TAcc, int, 1> const rw2,
            alpakaex::BufferAccessor<TAcc, int, 1, alpakaex::ReadWriteAccess> const rw3,
            alpakaex::BufferAccessor<TAcc, int, 1, alpakaex::ReadWriteAccess, TIdx> const rw4) const noexcept
        {
            static_assert(std::is_same_v<decltype(rw1), decltype(rw2)>);
            static_assert(std::is_same_v<decltype(rw2), decltype(rw3)>);
            static_assert(std::is_same_v<decltype(rw3), decltype(rw4)>);
        }
    };
} // namespace

TEST_CASE("BufferAccessor", "[accessor]")
{
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    using DevAcc = alpaka::Dev<Acc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;

    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto queue = Queue{devAcc};
    auto buffer = alpaka::allocBuf<int, Size>(devAcc, Size{1});

    auto const workdiv = alpaka::WorkDivMembers<Dim, Size>{
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}}};
    alpaka::exec<Acc>(
        queue,
        workdiv,
        BufferAccessorKernelRead{},
        alpakaex::readAccess(buffer),
        alpakaex::readAccess(buffer),
        alpakaex::readAccess(buffer));
    alpaka::exec<Acc>(
        queue,
        workdiv,
        BufferAccessorKernelWrite{},
        alpakaex::writeAccess(buffer),
        alpakaex::writeAccess(buffer),
        alpakaex::writeAccess(buffer));
    alpaka::exec<Acc>(
        queue,
        workdiv,
        BufferAccessorKernelReadWrite{},
        alpakaex::access(buffer),
        alpakaex::access(buffer),
        alpakaex::access(buffer),
        alpakaex::access(buffer));
}
