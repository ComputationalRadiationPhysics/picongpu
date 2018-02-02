/**
 * \file
 * Copyright 2015 Benjamin Worpitz
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

#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/push_front.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/clear.hpp>
#include <boost/mpl/pop_back.hpp>
#include <boost/mpl/contains.hpp>

#include <boost/predef.h>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

#include <iostream>
#include <type_traits>
#include <typeinfo>

BOOST_AUTO_TEST_SUITE(meta)

//-----------------------------------------------------------------------------
// This code is based on:
// http://stackoverflow.com/questions/5099429/how-to-use-stdtuple-types-with-boostmpl-algorithms/15865204#15865204
BOOST_AUTO_TEST_CASE(stdTupleAsMplSequence)
{
    using Tuple = std::tuple<int, char, bool>;

    static_assert(
        std::is_same<boost::mpl::front<Tuple>::type, int>::value,
        "boost::mpl::front on the std::tuple failed!");
    static_assert(
        boost::mpl::size<Tuple>::type::value == 3,
        "boost::mpl::size on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::pop_front<Tuple>::type, std::tuple<char, bool>>::value,
        "boost::mpl::pop_front on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::push_front<Tuple, unsigned>::type, std::tuple<unsigned, int, char, bool>>::value,
        "boost::mpl::push_front on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::push_back<Tuple, unsigned>::type, std::tuple<int, char, bool, unsigned>>::value,
        "boost::mpl::push_back on the std::tuple failed!");
    static_assert(
        boost::mpl::empty<Tuple>::type::value == false,
        "boost::mpl::empty on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::at_c<Tuple, 0>::type, int>::value,
        "boost::mpl::at_c on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::at_c<Tuple, 1>::type, char>::value,
        "boost::mpl::at_c on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::back<Tuple>::type, bool>::value,
        "boost::mpl::back on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::clear<Tuple>::type, std::tuple<>>::value,
        "boost::mpl::clear on the std::tuple failed!");
    static_assert(
        std::is_same<boost::mpl::pop_back<Tuple>::type, std::tuple<int, char>>::value,
        "boost::mpl::pop_back on the std::tuple failed!");
    static_assert(
        boost::mpl::contains<Tuple, int>::value,
        "boost::mpl::contains on the std::tuple failed!");
    static_assert(
        boost::mpl::contains<Tuple, char>::value,
        "boost::mpl::contains on the std::tuple failed!");
    static_assert(
        boost::mpl::contains<Tuple, bool>::value,
        "boost::mpl::contains on the std::tuple failed!");
    static_assert(
        boost::mpl::contains<Tuple, unsigned>::value == false,
        "boost::mpl::contains on the std::tuple failed!");
}

using TestTuple = std::tuple<int, char, bool>;

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    stdTupleAsMplSequenceTemplateTest,
    T,
    TestTuple)
{
    std::cout << typeid(T).name() << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
