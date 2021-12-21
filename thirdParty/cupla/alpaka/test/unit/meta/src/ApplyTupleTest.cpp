/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/ApplyTuple.hpp>

#include <catch2/catch.hpp>

//#############################################################################
struct Foo
{
    Foo(int num) : num_(num)
    {
    }
    auto add(int i) const -> int
    {
        return num_ + i;
    }
    int num_;
};

//-----------------------------------------------------------------------------
auto abs_num(int i) -> int;
auto abs_num(int i) -> int
{
    return std::abs(i);
}

//#############################################################################
struct AbsNum
{
    auto operator()(int i) const -> int
    {
        return std::abs(i);
    }
};

//-----------------------------------------------------------------------------
TEST_CASE("invoke", "[meta]")
{
    // invoke a free function
    REQUIRE(9 == alpaka::meta::invoke(abs_num, -9));

    // invoke a lambda
    REQUIRE(42 == alpaka::meta::invoke([]() { return abs_num(-42); }));

    // invoke a member function
    const Foo foo(-314159);
    REQUIRE(-314158 == alpaka::meta::invoke(&Foo::add, foo, 1));

    // invoke (access) a data member
    REQUIRE(-314159 == alpaka::meta::invoke(&Foo::num_, foo));

    // invoke a function object
    REQUIRE(18 == alpaka::meta::invoke(AbsNum(), -18));
}

//-----------------------------------------------------------------------------
auto add(int first, int second) -> int;
auto add(int first, int second) -> int
{
    return first + second;
}

//-----------------------------------------------------------------------------
template<typename T>
T add_generic(T first, T second)
{
    return first + second;
}

//-----------------------------------------------------------------------------
TEST_CASE("applyTuple", "[meta]")
{
    REQUIRE(3 == alpaka::meta::apply(add, std::make_tuple(1, 2)));

    // intended compilation error: template argument deduction/substitution fails
    // REQUIRE(5.0f == alpaka::meta::apply(add_generic, std::make_tuple(2.0f, 3.0f)));
}
