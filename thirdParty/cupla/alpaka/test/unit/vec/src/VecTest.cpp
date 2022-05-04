/* Copyright 2021 Axel Huebl, Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/test/dim/TestDims.hpp>
#include <alpaka/vec/Vec.hpp>

#include <catch2/catch.hpp>

#include <numeric>
#include <utility>

namespace
{
    namespace detail
    {
        template<typename F, std::size_t... Is>
        constexpr void foreachImpl(F&& f, std::index_sequence<Is...>)
        {
            (std::forward<F>(f)(std::integral_constant<std::size_t, Is>{}), ...);
        }
    } // namespace detail

    template<std::size_t I, typename F>
    constexpr void foreach(F&& f)
    {
        detail::foreachImpl(std::forward<F>(f), std::make_index_sequence<I>{});
    }
} // namespace

TEST_CASE("basicVecTraits", "[vec]")
{
    using Dim = alpaka::DimInt<3u>;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;

    STATIC_REQUIRE(std::is_trivially_copyable_v<Vec>);

    // constructor from Idx
    static constexpr Vec vec(static_cast<Idx>(0u), static_cast<Idx>(8u), static_cast<Idx>(15u));

    // constructor from convertible integral
    {
        [[maybe_unused]] constexpr Vec v0(0, 0, 0);
        [[maybe_unused]] constexpr Vec v1(1, 1, 1);
        [[maybe_unused]] constexpr Vec v2(1, 1u, 1);
        [[maybe_unused]] constexpr Vec v3(1, 1u, static_cast<Idx>(1u));
        [[maybe_unused]] constexpr Vec v4(1, 1u, 1.4f);
    }

    // constructor from convertible type
    {
        constexpr struct S
        {
            constexpr operator Idx() const
            {
                return 5;
            }
        } s;
        STATIC_REQUIRE(std::is_convertible_v<S, Idx>);

        [[maybe_unused]] constexpr Vec v(s, s, s);
    }

    // alpaka::Vec<0> default ctor
    {
        using Dim0 = alpaka::DimInt<0u>;
        [[maybe_unused]] alpaka::Vec<Dim0, Idx> const v1;
        [[maybe_unused]] constexpr alpaka::Vec<Dim0, Idx> v2{};
    }

    // default ctor
    {
        [[maybe_unused]] alpaka::Vec<Dim, Idx> const v1;
        [[maybe_unused]] constexpr alpaka::Vec<Dim, Idx> v2{};
    }

    // alpaka::subVecFromIndices
    {
        using IdxSequence = std::integer_sequence<std::size_t, 0u, Dim::value - 1u, 0u>;
        constexpr auto vecSubIndices(alpaka::subVecFromIndices<IdxSequence>(vec));

        STATIC_REQUIRE(vecSubIndices[0u] == vec[0u]);
        STATIC_REQUIRE(vecSubIndices[1u] == vec[Dim::value - 1u]);
        STATIC_REQUIRE(vecSubIndices[2u] == vec[0u]);
    }

    // alpaka::subVecBegin
    {
        using DimSubVecEnd = alpaka::DimInt<2u>;
        static constexpr auto vecSubBegin(alpaka::subVecBegin<DimSubVecEnd>(vec));

        foreach
            <DimSubVecEnd::value>(
                [&](auto ic)
                {
                    constexpr auto i = decltype(ic)::value;
                    STATIC_REQUIRE(vecSubBegin[i] == vec[i]);
                });
    }

    // alpaka::subVecEnd
    {
        using DimSubVecEnd = alpaka::DimInt<2u>;
        static constexpr auto vecSubEnd(alpaka::subVecEnd<DimSubVecEnd>(vec));

        foreach
            <DimSubVecEnd::value>(
                [&](auto ic)
                {
                    constexpr auto i = decltype(ic)::value;
                    STATIC_REQUIRE(vecSubEnd[i] == vec[Dim::value - DimSubVecEnd::value + i]);
                });
    }

    // alpaka::castVec
    {
        using SizeCast = std::uint16_t;
        static constexpr auto vecCast(alpaka::castVec<SizeCast>(vec));

        /*using VecCastConst = decltype(vecCast);
        using VecCast = std::decay_t<VecCastConst>;
        STATIC_REQUIRE(
            std::is_same_v<
                alpaka::Idx<VecCast>,
                SizeCast
            >);*/

        foreach
            <Dim::value>(
                [&](auto ic)
                {
                    constexpr auto i = decltype(ic)::value;
                    STATIC_REQUIRE(vecCast[i] == static_cast<SizeCast>(vec[i]));
                });
    }

    // alpaka::reverseVec
    {
        static constexpr auto vecReverse(alpaka::reverseVec(vec));

        foreach
            <Dim::value>(
                [&](auto ic)
                {
                    constexpr auto i = decltype(ic)::value;
                    STATIC_REQUIRE(vecReverse[i] == vec[Dim::value - 1u - i]);
                });
    }

    // alpaka::concatVec
    {
        using Dim2 = alpaka::DimInt<2u>;
        static constexpr alpaka::Vec<Dim2, Idx> vec2(static_cast<Idx>(47u), static_cast<Idx>(11u));

        static constexpr auto vecConcat(alpaka::concatVec(vec, vec2));
        STATIC_REQUIRE(std::is_same_v<alpaka::Dim<std::decay_t<decltype(vecConcat)>>, alpaka::DimInt<5u>>);

        foreach
            <Dim::value>(
                [&](auto ic)
                {
                    constexpr auto i = decltype(ic)::value;
                    STATIC_REQUIRE(vecConcat[i] == vec[i]);
                });
        foreach
            <Dim2::value>(
                [&](auto ic)
                {
                    constexpr auto i = decltype(ic)::value;
                    STATIC_REQUIRE(vecConcat[Dim::value + i] == vec2[i]);
                });
    }

    {
        constexpr alpaka::Vec<Dim, Idx> vec3(static_cast<Idx>(47u), static_cast<Idx>(8u), static_cast<Idx>(3u));

        // alpaka::Vec operator +
        {
            constexpr auto vecLessEqual(vec + vec3);
            STATIC_REQUIRE(std::is_same_v<alpaka::Dim<std::decay_t<decltype(vecLessEqual)>>, Dim>);
            STATIC_REQUIRE(std::is_same_v<alpaka::Idx<std::decay_t<decltype(vecLessEqual)>>, Idx>);

            constexpr alpaka::Vec<Dim, Idx> referenceVec(
                static_cast<Idx>(47u),
                static_cast<Idx>(16u),
                static_cast<Idx>(18u));
            STATIC_REQUIRE(referenceVec == vecLessEqual);
        }

        // alpaka::Vec operator -
        {
            constexpr auto vecLessEqual(vec - vec3);
            STATIC_REQUIRE(std::is_same_v<alpaka::Dim<std::decay_t<decltype(vecLessEqual)>>, Dim>);
            STATIC_REQUIRE(std::is_same_v<alpaka::Idx<std::decay_t<decltype(vecLessEqual)>>, Idx>);

            constexpr alpaka::Vec<Dim, Idx> referenceVec(
                static_cast<Idx>(-47),
                static_cast<Idx>(0u),
                static_cast<Idx>(12u));
            STATIC_REQUIRE(referenceVec == vecLessEqual);
        }

        // alpaka::Vec operator *
        {
            constexpr auto vecLessEqual(vec * vec3);
            STATIC_REQUIRE(std::is_same_v<alpaka::Dim<std::decay_t<decltype(vecLessEqual)>>, Dim>);
            STATIC_REQUIRE(std::is_same_v<alpaka::Idx<std::decay_t<decltype(vecLessEqual)>>, Idx>);

            constexpr alpaka::Vec<Dim, Idx> referenceVec(
                static_cast<Idx>(0u),
                static_cast<Idx>(64u),
                static_cast<Idx>(45u));
            STATIC_REQUIRE(referenceVec == vecLessEqual);
        }

        // alpaka::Vec operator <
        {
            constexpr auto vecLessEqual(vec < vec3);
            STATIC_REQUIRE(std::is_same_v<alpaka::Dim<std::decay_t<decltype(vecLessEqual)>>, Dim>);
            STATIC_REQUIRE(std::is_same_v<alpaka::Idx<std::decay_t<decltype(vecLessEqual)>>, bool>);

            constexpr alpaka::Vec<Dim, bool> referenceVec(true, false, false);
            STATIC_REQUIRE(referenceVec == vecLessEqual);
        }

        // alpaka::Vec operator <=
        {
            constexpr auto vecLessEqual(vec <= vec3);
            STATIC_REQUIRE(std::is_same_v<alpaka::Dim<std::decay_t<decltype(vecLessEqual)>>, Dim>);
            STATIC_REQUIRE(std::is_same_v<alpaka::Idx<std::decay_t<decltype(vecLessEqual)>>, bool>);

            constexpr alpaka::Vec<Dim, bool> referenceVec(true, true, false);
            STATIC_REQUIRE(referenceVec == vecLessEqual);
        }

        // alpaka::Vec operator >=
        {
            constexpr auto vecLessEqual(vec >= vec3);
            STATIC_REQUIRE(std::is_same_v<alpaka::Dim<std::decay_t<decltype(vecLessEqual)>>, Dim>);
            STATIC_REQUIRE(std::is_same_v<alpaka::Idx<std::decay_t<decltype(vecLessEqual)>>, bool>);

            constexpr alpaka::Vec<Dim, bool> referenceVec(false, true, true);

            STATIC_REQUIRE(referenceVec == vecLessEqual);
        }

        // alpaka::Vec operator >
        {
            constexpr auto vecLessEqual(vec > vec3);
            STATIC_REQUIRE(std::is_same_v<alpaka::Dim<std::decay_t<decltype(vecLessEqual)>>, Dim>);
            STATIC_REQUIRE(std::is_same_v<alpaka::Idx<std::decay_t<decltype(vecLessEqual)>>, bool>);

            constexpr alpaka::Vec<Dim, bool> referenceVec(false, false, true);
            STATIC_REQUIRE(referenceVec == vecLessEqual);
        }

        // alpaka::Vec begin/end
        STATIC_REQUIRE(
            []
            {
                auto v = alpaka::Vec<Dim, int>::ones();
                for(auto& e : v)
                {
                    int i = e; // read
                    e += i; // write
                }
                return v == alpaka::Vec<Dim, int>::all(2);
            }());

        // const alpaka::Vec begin/end
        STATIC_REQUIRE(
            []
            {
                const auto v = alpaka::Vec<Dim, int>::ones();
                int sum = 0;
                for(const auto& e : v)
                    sum += e; // read
                return sum == Dim::value;
            }());
    }
}

template<typename TDim, typename TIdx>
struct NonAlpakaVec
{
    operator ::alpaka::Vec<TDim, TIdx>() const
    {
        using AlpakaVector = ::alpaka::Vec<TDim, TIdx>;
        AlpakaVector result = AlpakaVector::zeros();

        if constexpr(TDim::value > 0)
        {
            for(TIdx d(0); d < TDim::value; ++d)
            {
                result[TDim::value - 1 - d] = (*this)[d];
            }
        }

        return result;
    }
    auto operator[](TIdx /*idx*/) const -> TIdx
    {
        return static_cast<TIdx>(0);
    }
};

TEMPLATE_LIST_TEST_CASE("vecNDConstructionFromNonAlpakaVec", "[vec]", alpaka::test::TestDims)
{
    using Dim = TestType;
    if constexpr(Dim::value > 0)
    {
        using Idx = std::size_t;
        auto const nonAlpakaVec = NonAlpakaVec<Dim, Idx>{};
        auto const alpakaVec = static_cast<alpaka::Vec<Dim, Idx>>(nonAlpakaVec);

        for(Idx d(0); d < Dim::value; ++d)
        {
            REQUIRE(nonAlpakaVec[d] == alpakaVec[d]);
        }
    }
}

TEST_CASE("structuredBindings", "[vec]")
{
    using Dim = alpaka::DimInt<2u>;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;

    Vec vec{Idx{1}, Idx{2}};

    const auto [y, x] = vec;
    CHECK(y == 1);
    CHECK(x == 2);

    auto& [y2, x2] = vec;
    y2++;
    x2++;
    CHECK(vec[0] == 2);
    CHECK(vec[1] == 3);
}
