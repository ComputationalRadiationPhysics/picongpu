/**
 * \file
 * Copyright 2016 Erik Zenker
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

// \Hack: Boost.MPL defines BOOST_MPL_CFG_GPU_ENABLED to __host__ __device__ if nvcc is used.
// BOOST_AUTO_TEST_CASE_TEMPLATE and its internals are not GPU enabled but is using boost::mpl::for_each internally.
// For each template parameter this leads to:
// /home/travis/build/boost/boost/mpl/for_each.hpp(78): warning: calling a __host__ function from a __host__ __device__ function is not allowed
// because boost::mpl::for_each has the BOOST_MPL_CFG_GPU_ENABLED attribute but the test internals are pure host methods.
// Because we do not use MPL within GPU code here, we can disable the MPL GPU support.
#define BOOST_MPL_CFG_GPU_ENABLED

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>

#include <alpaka/core/BoostPredef.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

BOOST_AUTO_TEST_SUITE(vec)

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
    basicVecTraits)
{
    using Dim = alpaka::dim::DimInt<3u>;
    using Idx = std::size_t;
    using Vec = alpaka::vec::Vec<Dim, Idx>;

    Vec const vec(
        static_cast<Idx>(0u),
        static_cast<Idx>(8u),
        static_cast<Idx>(15u));



    //-----------------------------------------------------------------------------
    // alpaka::vec::Vec zero elements
    {
        using Dim0 = alpaka::dim::DimInt<0u>;
        alpaka::vec::Vec<Dim0, Idx> const vec0{};
    }

    //-----------------------------------------------------------------------------
    // alpaka::vec::subVecFromIndices
    {
        using IdxSequence =
            alpaka::meta::IntegerSequence<
                std::size_t,
                0u,
                Dim::value -1u,
                0u>;
        auto const vecSubIndices(
            alpaka::vec::subVecFromIndices<
                IdxSequence>(
                    vec));

        BOOST_REQUIRE_EQUAL(vecSubIndices[0u], vec[0u]);
        BOOST_REQUIRE_EQUAL(vecSubIndices[1u], vec[Dim::value -1u]);
        BOOST_REQUIRE_EQUAL(vecSubIndices[2u], vec[0u]);
    }

    //-----------------------------------------------------------------------------
    // alpaka::vec::subVecBegin
    {
        using DimSubVecEnd =
            alpaka::dim::DimInt<2u>;
        auto const vecSubBegin(
            alpaka::vec::subVecBegin<
                DimSubVecEnd>(
                    vec));

        for(typename Dim::value_type i(0); i < DimSubVecEnd::value; ++i)
        {
            BOOST_REQUIRE_EQUAL(vecSubBegin[i], vec[i]);
        }
    }

    //-----------------------------------------------------------------------------
    // alpaka::vec::subVecEnd
    {
        using DimSubVecEnd =
            alpaka::dim::DimInt<2u>;
        auto const vecSubEnd(
            alpaka::vec::subVecEnd<
                DimSubVecEnd>(
                    vec));

        for(typename Dim::value_type i(0); i < DimSubVecEnd::value; ++i)
        {
            BOOST_REQUIRE_EQUAL(vecSubEnd[i], vec[Dim::value - DimSubVecEnd::value + i]);
        }
    }

    //-----------------------------------------------------------------------------
    // alpaka::vec::cast
    {
        using SizeCast = std::uint16_t;
        auto const vecCast(
            alpaka::vec::cast<
                SizeCast>(
                    vec));

        /*using VecCastConst = decltype(vecCast);
        using VecCast = typename std::decay<VecCastConst>::type;
        static_assert(
            std::is_same<
                alpaka::idx::Idx<VecCast>,
                SizeCast
            >::value,
            "The idx type of the casted vec is wrong");*/

        for(typename Dim::value_type i(0); i < Dim::value; ++i)
        {
            BOOST_REQUIRE_EQUAL(vecCast[i], static_cast<SizeCast>(vec[i]));
        }
    }

    //-----------------------------------------------------------------------------
    // alpaka::vec::reverse
    {
        auto const vecReverse(
            alpaka::vec::reverse(
                vec));

        for(typename Dim::value_type i(0); i < Dim::value; ++i)
        {
            BOOST_REQUIRE_EQUAL(vecReverse[i], vec[Dim::value - 1u - i]);
        }
    }

    //-----------------------------------------------------------------------------
    // alpaka::vec::concat
    {
        using Dim2 = alpaka::dim::DimInt<2u>;
        alpaka::vec::Vec<Dim2, Idx> const vec2(
            static_cast<Idx>(47u),
            static_cast<Idx>(11u));

        auto const vecConcat(
            alpaka::vec::concat(
                vec,
                vec2));

        static_assert(
            std::is_same<alpaka::dim::Dim<std::decay<decltype(vecConcat)>::type>, alpaka::dim::DimInt<5u>>::value,
            "Result dimension type of concatenation incorrect!");

        for(typename Dim::value_type i(0); i < Dim::value; ++i)
        {
            BOOST_REQUIRE_EQUAL(vecConcat[i], vec[i]);
        }
        for(typename Dim2::value_type i(0); i < Dim2::value; ++i)
        {
            BOOST_REQUIRE_EQUAL(vecConcat[Dim::value + i], vec2[i]);
        }
    }

    {
        alpaka::vec::Vec<Dim, Idx> const vec3(
            static_cast<Idx>(47u),
            static_cast<Idx>(8u),
            static_cast<Idx>(3u));

        //-----------------------------------------------------------------------------
        // alpaka::vec::Vec operator +
        {
            auto const vecLessEqual(vec + vec3);

            static_assert(
                std::is_same<alpaka::dim::Dim<std::decay<decltype(vecLessEqual)>::type>, Dim>::value,
                "Result dimension type of operator <= incorrect!");

            static_assert(
                std::is_same<alpaka::idx::Idx<std::decay<decltype(vecLessEqual)>::type>, Idx>::value,
                "Result idx type of operator <= incorrect!");

            alpaka::vec::Vec<Dim, Idx> const referenceVec(
                static_cast<Idx>(47u),
                static_cast<Idx>(16u),
                static_cast<Idx>(18u));

            BOOST_REQUIRE_EQUAL(referenceVec, vecLessEqual);
        }

        //-----------------------------------------------------------------------------
        // alpaka::vec::Vec operator -
        {
            auto const vecLessEqual(vec - vec3);

            static_assert(
                std::is_same<alpaka::dim::Dim<std::decay<decltype(vecLessEqual)>::type>, Dim>::value,
                "Result dimension type of operator <= incorrect!");

            static_assert(
                std::is_same<alpaka::idx::Idx<std::decay<decltype(vecLessEqual)>::type>, Idx>::value,
                "Result idx type of operator <= incorrect!");

            alpaka::vec::Vec<Dim, Idx> const referenceVec(
                static_cast<Idx>(-47),
                static_cast<Idx>(0u),
                static_cast<Idx>(12u));

            BOOST_REQUIRE_EQUAL(referenceVec, vecLessEqual);
        }

        //-----------------------------------------------------------------------------
        // alpaka::vec::Vec operator *
        {
            auto const vecLessEqual(vec * vec3);

            static_assert(
                std::is_same<alpaka::dim::Dim<std::decay<decltype(vecLessEqual)>::type>, Dim>::value,
                "Result dimension type of operator <= incorrect!");

            static_assert(
                std::is_same<alpaka::idx::Idx<std::decay<decltype(vecLessEqual)>::type>, Idx>::value,
                "Result idx type of operator <= incorrect!");

            alpaka::vec::Vec<Dim, Idx> const referenceVec(
                static_cast<Idx>(0u),
                static_cast<Idx>(64u),
                static_cast<Idx>(45u));

            BOOST_REQUIRE_EQUAL(referenceVec, vecLessEqual);
        }

        //-----------------------------------------------------------------------------
        // alpaka::vec::Vec operator <
        {
            auto const vecLessEqual(vec < vec3);

            static_assert(
                std::is_same<alpaka::dim::Dim<std::decay<decltype(vecLessEqual)>::type>, Dim>::value,
                "Result dimension type of operator <= incorrect!");

            static_assert(
                std::is_same<alpaka::idx::Idx<std::decay<decltype(vecLessEqual)>::type>, bool>::value,
                "Result idx type of operator <= incorrect!");

            alpaka::vec::Vec<Dim, bool> const referenceVec(
                true,
                false,
                false);

            BOOST_REQUIRE_EQUAL(referenceVec, vecLessEqual);
        }

        //-----------------------------------------------------------------------------
        // alpaka::vec::Vec operator <=
        {
            auto const vecLessEqual(vec <= vec3);

            static_assert(
                std::is_same<alpaka::dim::Dim<std::decay<decltype(vecLessEqual)>::type>, Dim>::value,
                "Result dimension type of operator <= incorrect!");

            static_assert(
                std::is_same<alpaka::idx::Idx<std::decay<decltype(vecLessEqual)>::type>, bool>::value,
                "Result idx type of operator <= incorrect!");

            alpaka::vec::Vec<Dim, bool> const referenceVec(
                true,
                true,
                false);

            BOOST_REQUIRE_EQUAL(referenceVec, vecLessEqual);
        }

        //-----------------------------------------------------------------------------
        // alpaka::vec::Vec operator >=
        {
            auto const vecLessEqual(vec >= vec3);

            static_assert(
                std::is_same<alpaka::dim::Dim<std::decay<decltype(vecLessEqual)>::type>, Dim>::value,
                "Result dimension type of operator <= incorrect!");

            static_assert(
                std::is_same<alpaka::idx::Idx<std::decay<decltype(vecLessEqual)>::type>, bool>::value,
                "Result idx type of operator <= incorrect!");

            alpaka::vec::Vec<Dim, bool> const referenceVec(
                false,
                true,
                true);

            BOOST_REQUIRE_EQUAL(referenceVec, vecLessEqual);
        }

        //-----------------------------------------------------------------------------
        // alpaka::vec::Vec operator >
        {
            auto const vecLessEqual(vec > vec3);

            static_assert(
                std::is_same<alpaka::dim::Dim<std::decay<decltype(vecLessEqual)>::type>, Dim>::value,
                "Result dimension type of operator <= incorrect!");

            static_assert(
                std::is_same<alpaka::idx::Idx<std::decay<decltype(vecLessEqual)>::type>, bool>::value,
                "Result idx type of operator <= incorrect!");

            alpaka::vec::Vec<Dim, bool> const referenceVec(
                false,
                false,
                true);

            BOOST_REQUIRE_EQUAL(referenceVec, vecLessEqual);
        }
    }
}

//#############################################################################
template<
    typename TDim,
    typename TIdx>
struct NonAlpakaVec
{
    //-----------------------------------------------------------------------------
    operator ::alpaka::vec::Vec<
        TDim,
        TIdx>() const
    {
        using AlpakaVector = ::alpaka::vec::Vec<
            TDim,
            TIdx
        >;
        AlpakaVector result(AlpakaVector::zeros());

        for(TIdx d(0); d < TDim::value; ++d)
        {
            result[TDim::value - 1 - d] = (*this)[d];
        }

        return result;
    }
    //-----------------------------------------------------------------------------
    auto operator [](TIdx /*idx*/) const
    -> TIdx
    {
        return static_cast<TIdx>(0);
    }
};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    vecNDConstructionFromNonAlpakaVec,
    TDim,
    alpaka::test::acc::TestDims)
{
    using Idx = std::size_t;

    NonAlpakaVec<TDim, Idx> nonAlpakaVec;
    auto const alpakaVec(static_cast<alpaka::vec::Vec<TDim, Idx>>(nonAlpakaVec));

    for(Idx d(0); d < TDim::value; ++d)
    {
        BOOST_REQUIRE_EQUAL(nonAlpakaVec[d], alpakaVec[d]);
    }
}

BOOST_AUTO_TEST_SUITE_END()
