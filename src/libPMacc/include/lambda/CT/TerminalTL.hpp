/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "../ExprTypes.h"
#include "../placeholder.h"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/front_inserter.hpp>
#include "../Expression.hpp"
#include <lambda/ProxyClass.hpp>

namespace mpl = boost::mpl;

namespace PMacc
{
namespace lambda
{
namespace CT
{

struct Type_is_not_defined {};

template<typename Expr>
struct TerminalTL {typedef Type_is_not_defined type;};

template<>
struct TerminalTL<mpl::void_>
{
    typedef mpl::vector<> type;
};

template<typename Child0>
struct TerminalTL<Expression<exprTypes::terminal, mpl::vector<Child0> > >
{
    typedef mpl::vector<ProxyClass<Child0> > type;
};

template<int I>
struct TerminalTL<Expression<exprTypes::terminal, mpl::vector<placeholder<I> > > >
{
    typedef mpl::vector<> type;
};

template<int I>
struct TerminalTL<Expression<exprTypes::terminal, mpl::vector<mpl::int_<I> > > >
{
    typedef mpl::vector<> type;
};

template<typename Child0, typename Child1>
struct TerminalTL<Expression<exprTypes::assign, mpl::vector<Child0, Child1> > >
{
    typedef typename mpl::reverse_copy<typename TerminalTL<Child0>::type,
                   mpl::front_inserter<typename TerminalTL<Child1>::type> >::type type;
};

template<typename Child0, typename Child1>
struct TerminalTL<Expression<exprTypes::plus, mpl::vector<Child0, Child1> > >
{
    typedef typename mpl::reverse_copy<typename TerminalTL<Child0>::type,
                               mpl::front_inserter<typename TerminalTL<Child1>::type> >::type type;
};

template<typename Child0, typename Child1>
struct TerminalTL<Expression<exprTypes::minus, mpl::vector<Child0, Child1> > >
{
    typedef typename mpl::reverse_copy<typename TerminalTL<Child0>::type,
                               mpl::front_inserter<typename TerminalTL<Child1>::type> >::type type;
};

template<typename Child0, typename Child1>
struct TerminalTL<Expression<exprTypes::multiply, mpl::vector<Child0, Child1> > >
{
    typedef typename mpl::reverse_copy<typename TerminalTL<Child0>::type,
                               mpl::front_inserter<typename TerminalTL<Child1>::type> >::type type;
};

template<typename Child0, typename Child1>
struct TerminalTL<Expression<exprTypes::divide, mpl::vector<Child0, Child1> > >
{
    typedef typename mpl::reverse_copy<typename TerminalTL<Child0>::type,
                               mpl::front_inserter<typename TerminalTL<Child1>::type> >::type type;
};

template<typename Child0, typename Child1>
struct TerminalTL<Expression<exprTypes::comma, mpl::vector<Child0, Child1> > >
{
    typedef typename mpl::reverse_copy<typename TerminalTL<Child0>::type,
                               mpl::front_inserter<typename TerminalTL<Child1>::type> >::type type;
};

#define COMBINE(x, y) typename mpl::reverse_copy< x, \
                     mpl::front_inserter< y > >::type


//\todo: eine call Spezialisierung reicht
template<typename Child0>
struct TerminalTL<Expression<exprTypes::call, mpl::vector<Child0> > >
{
    typedef typename TerminalTL<Child0>::type type;
};

template<typename Child0, typename Child1>
struct TerminalTL<Expression<exprTypes::call, mpl::vector<Child0, Child1> > >
{
    typedef COMBINE(typename TerminalTL<Child0>::type,
                    typename TerminalTL<Child1>::type) type;
};

template<typename Child0, typename Child1, typename Child2>
struct TerminalTL<Expression<exprTypes::call, mpl::vector<Child0, Child1, Child2> > >
{
    typedef COMBINE(
            COMBINE(typename TerminalTL<Child0>::type,
                    typename TerminalTL<Child1>::type),
                    typename TerminalTL<Child2>::type) type;
};

template<typename Child0, typename Child1, typename Child2, typename Child3>
struct TerminalTL<Expression<exprTypes::call, mpl::vector<Child0, Child1, Child2, Child3> > >
{
    typedef COMBINE(
            COMBINE(
            COMBINE(typename TerminalTL<Child0>::type,
                    typename TerminalTL<Child1>::type),
                    typename TerminalTL<Child2>::type),
                    typename TerminalTL<Child3>::type) type;
};

template<typename Child0, typename Child1, typename Child2, typename Child3, typename Child4>
struct TerminalTL<Expression<exprTypes::call, mpl::vector<Child0, Child1, Child2, Child3, Child4> > >
{
    typedef COMBINE(
            COMBINE(
            COMBINE(
            COMBINE(typename TerminalTL<Child0>::type,
                    typename TerminalTL<Child1>::type),
                    typename TerminalTL<Child2>::type),
                    typename TerminalTL<Child3>::type),
                    typename TerminalTL<Child4>::type) type;
};

template<typename Child0, typename Child1, typename Child2, typename Child3, typename Child4, typename Child5>
struct TerminalTL<Expression<exprTypes::call, mpl::vector<Child0, Child1, Child2, Child3, Child4, Child5> > >
{
    typedef COMBINE(
            COMBINE(
            COMBINE(
            COMBINE(
            COMBINE(typename TerminalTL<Child0>::type,
                    typename TerminalTL<Child1>::type),
                    typename TerminalTL<Child2>::type),
                    typename TerminalTL<Child3>::type),
                    typename TerminalTL<Child4>::type),
                    typename TerminalTL<Child5>::type) type;
};

template<typename Child0, typename Child1, typename Child2, typename Child3, typename Child4, typename Child5, typename Child6>
struct TerminalTL<Expression<exprTypes::call, mpl::vector<Child0, Child1, Child2, Child3, Child4, Child5, Child6> > >
{
    typedef COMBINE(
            COMBINE(
            COMBINE(
            COMBINE(
            COMBINE(
            COMBINE(typename TerminalTL<Child0>::type,
                    typename TerminalTL<Child1>::type),
                    typename TerminalTL<Child2>::type),
                    typename TerminalTL<Child3>::type),
                    typename TerminalTL<Child4>::type),
                    typename TerminalTL<Child5>::type),
                    typename TerminalTL<Child6>::type) type;
};

#undef COMBINE

template<typename Child0, typename Child1>
struct TerminalTL<Expression<exprTypes::subscript, mpl::vector<Child0, Child1> > >
{
    typedef typename mpl::reverse_copy<typename TerminalTL<Child0>::type,
                               mpl::front_inserter<typename TerminalTL<Child1>::type> >::type type;
};

} // CT
} // lambda
} // PMacc

