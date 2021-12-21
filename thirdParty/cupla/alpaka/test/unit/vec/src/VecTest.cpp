/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Erik Zenker
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

#include <utility>

//-----------------------------------------------------------------------------
TEST_CASE("basicVecTraits", "[vec]")
{
    using Dim = alpaka::DimInt<3u>;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;

    Vec const vec(static_cast<Idx>(0u), static_cast<Idx>(8u), static_cast<Idx>(15u));


    //-----------------------------------------------------------------------------
    // alpaka::Vec zero elements
    {
        using Dim0 = alpaka::DimInt<0u>;
        alpaka::Vec<Dim0, Idx> const vec0{};
    }

    //-----------------------------------------------------------------------------
    // alpaka::subVecFromIndices
    {
        using IdxSequence = std::integer_sequence<std::size_t, 0u, Dim::value - 1u, 0u>;
        auto const vecSubIndices(alpaka::subVecFromIndices<IdxSequence>(vec));

        REQUIRE(vecSubIndices[0u] == vec[0u]);
        REQUIRE(vecSubIndices[1u] == vec[Dim::value - 1u]);
        REQUIRE(vecSubIndices[2u] == vec[0u]);
    }

    //-----------------------------------------------------------------------------
    // alpaka::subVecBegin
    {
        using DimSubVecEnd = alpaka::DimInt<2u>;
        auto const vecSubBegin(alpaka::subVecBegin<DimSubVecEnd>(vec));

        for(typename Dim::value_type i(0); i < DimSubVecEnd::value; ++i)
        {
            REQUIRE(vecSubBegin[i] == vec[i]);
        }
    }

    //-----------------------------------------------------------------------------
    // alpaka::subVecEnd
    {
        using DimSubVecEnd = alpaka::DimInt<2u>;
        auto const vecSubEnd(alpaka::subVecEnd<DimSubVecEnd>(vec));

        for(typename Dim::value_type i(0); i < DimSubVecEnd::value; ++i)
        {
            REQUIRE(vecSubEnd[i] == vec[Dim::value - DimSubVecEnd::value + i]);
        }
    }

    //-----------------------------------------------------------------------------
    // alpaka::castVec
    {
        using SizeCast = std::uint16_t;
        auto const vecCast(alpaka::castVec<SizeCast>(vec));

        /*using VecCastConst = decltype(vecCast);
        using VecCast = std::decay_t<VecCastConst>;
        static_assert(
            std::is_same<
                alpaka::Idx<VecCast>,
                SizeCast
            >::value,
            "The idx type of the casted vec is wrong");*/

        for(typename Dim::value_type i(0); i < Dim::value; ++i)
        {
            REQUIRE(vecCast[i] == static_cast<SizeCast>(vec[i]));
        }
    }

    //-----------------------------------------------------------------------------
    // alpaka::reverseVec
    {
        auto const vecReverse(alpaka::reverseVec(vec));

        for(typename Dim::value_type i(0); i < Dim::value; ++i)
        {
            REQUIRE(vecReverse[i] == vec[Dim::value - 1u - i]);
        }
    }

    //-----------------------------------------------------------------------------
    // alpaka::concatVec
    {
        using Dim2 = alpaka::DimInt<2u>;
        alpaka::Vec<Dim2, Idx> const vec2(static_cast<Idx>(47u), static_cast<Idx>(11u));

        auto const vecConcat(alpaka::concatVec(vec, vec2));

        static_assert(
            std::is_same<alpaka::Dim<std::decay<decltype(vecConcat)>::type>, alpaka::DimInt<5u>>::value,
            "Result dimension type of concatenation incorrect!");

        for(typename Dim::value_type i(0); i < Dim::value; ++i)
        {
            REQUIRE(vecConcat[i] == vec[i]);
        }
        for(typename Dim2::value_type i(0); i < Dim2::value; ++i)
        {
            REQUIRE(vecConcat[Dim::value + i] == vec2[i]);
        }
    }

    {
        alpaka::Vec<Dim, Idx> const vec3(static_cast<Idx>(47u), static_cast<Idx>(8u), static_cast<Idx>(3u));

        //-----------------------------------------------------------------------------
        // alpaka::Vec operator +
        {
            auto const vecLessEqual(vec + vec3);

            static_assert(
                std::is_same<alpaka::Dim<std::decay<decltype(vecLessEqual)>::type>, Dim>::value,
                "Result dimension type of operator <= incorrect!");

            static_assert(
                std::is_same<alpaka::Idx<std::decay<decltype(vecLessEqual)>::type>, Idx>::value,
                "Result idx type of operator <= incorrect!");

            alpaka::Vec<Dim, Idx> const referenceVec(
                static_cast<Idx>(47u),
                static_cast<Idx>(16u),
                static_cast<Idx>(18u));

            REQUIRE(referenceVec == vecLessEqual);
        }

        //-----------------------------------------------------------------------------
        // alpaka::Vec operator -
        {
            auto const vecLessEqual(vec - vec3);

            static_assert(
                std::is_same<alpaka::Dim<std::decay<decltype(vecLessEqual)>::type>, Dim>::value,
                "Result dimension type of operator <= incorrect!");

            static_assert(
                std::is_same<alpaka::Idx<std::decay<decltype(vecLessEqual)>::type>, Idx>::value,
                "Result idx type of operator <= incorrect!");

            alpaka::Vec<Dim, Idx> const referenceVec(
                static_cast<Idx>(-47),
                static_cast<Idx>(0u),
                static_cast<Idx>(12u));

            REQUIRE(referenceVec == vecLessEqual);
        }

        //-----------------------------------------------------------------------------
        // alpaka::Vec operator *
        {
            auto const vecLessEqual(vec * vec3);

            static_assert(
                std::is_same<alpaka::Dim<std::decay<decltype(vecLessEqual)>::type>, Dim>::value,
                "Result dimension type of operator <= incorrect!");

            static_assert(
                std::is_same<alpaka::Idx<std::decay<decltype(vecLessEqual)>::type>, Idx>::value,
                "Result idx type of operator <= incorrect!");

            alpaka::Vec<Dim, Idx> const referenceVec(
                static_cast<Idx>(0u),
                static_cast<Idx>(64u),
                static_cast<Idx>(45u));

            REQUIRE(referenceVec == vecLessEqual);
        }

        //-----------------------------------------------------------------------------
        // alpaka::Vec operator <
        {
            auto const vecLessEqual(vec < vec3);

            static_assert(
                std::is_same<alpaka::Dim<std::decay<decltype(vecLessEqual)>::type>, Dim>::value,
                "Result dimension type of operator <= incorrect!");

            static_assert(
                std::is_same<alpaka::Idx<std::decay<decltype(vecLessEqual)>::type>, bool>::value,
                "Result idx type of operator <= incorrect!");

            alpaka::Vec<Dim, bool> const referenceVec(true, false, false);

            REQUIRE(referenceVec == vecLessEqual);
        }

        //-----------------------------------------------------------------------------
        // alpaka::Vec operator <=
        {
            auto const vecLessEqual(vec <= vec3);

            static_assert(
                std::is_same<alpaka::Dim<std::decay<decltype(vecLessEqual)>::type>, Dim>::value,
                "Result dimension type of operator <= incorrect!");

            static_assert(
                std::is_same<alpaka::Idx<std::decay<decltype(vecLessEqual)>::type>, bool>::value,
                "Result idx type of operator <= incorrect!");

            alpaka::Vec<Dim, bool> const referenceVec(true, true, false);

            REQUIRE(referenceVec == vecLessEqual);
        }

        //-----------------------------------------------------------------------------
        // alpaka::Vec operator >=
        {
            auto const vecLessEqual(vec >= vec3);

            static_assert(
                std::is_same<alpaka::Dim<std::decay<decltype(vecLessEqual)>::type>, Dim>::value,
                "Result dimension type of operator <= incorrect!");

            static_assert(
                std::is_same<alpaka::Idx<std::decay<decltype(vecLessEqual)>::type>, bool>::value,
                "Result idx type of operator <= incorrect!");

            alpaka::Vec<Dim, bool> const referenceVec(false, true, true);

            REQUIRE(referenceVec == vecLessEqual);
        }

        //-----------------------------------------------------------------------------
        // alpaka::Vec operator >
        {
            auto const vecLessEqual(vec > vec3);

            static_assert(
                std::is_same<alpaka::Dim<std::decay<decltype(vecLessEqual)>::type>, Dim>::value,
                "Result dimension type of operator <= incorrect!");

            static_assert(
                std::is_same<alpaka::Idx<std::decay<decltype(vecLessEqual)>::type>, bool>::value,
                "Result idx type of operator <= incorrect!");

            alpaka::Vec<Dim, bool> const referenceVec(false, false, true);

            REQUIRE(referenceVec == vecLessEqual);
        }
    }
}

//#############################################################################
template<typename TDim, typename TIdx>
struct NonAlpakaVec
{
    //-----------------------------------------------------------------------------
    operator ::alpaka::Vec<TDim, TIdx>() const
    {
        using AlpakaVector = ::alpaka::Vec<TDim, TIdx>;
        AlpakaVector result(AlpakaVector::zeros());

        for(TIdx d(0); d < TDim::value; ++d)
        {
            result[TDim::value - 1 - d] = (*this)[d];
        }

        return result;
    }
    //-----------------------------------------------------------------------------
    auto operator[](TIdx /*idx*/) const -> TIdx
    {
        return static_cast<TIdx>(0);
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("vecNDConstructionFromNonAlpakaVec", "[vec]", alpaka::test::TestDims)
{
    using Dim = TestType;
    using Idx = std::size_t;

    NonAlpakaVec<Dim, Idx> nonAlpakaVec;
    auto const alpakaVec(static_cast<alpaka::Vec<Dim, Idx>>(nonAlpakaVec));

    for(Idx d(0); d < Dim::value; ++d)
    {
        REQUIRE(nonAlpakaVec[d] == alpakaVec[d]);
    }
}
