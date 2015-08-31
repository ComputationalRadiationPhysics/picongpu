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

#include "../Expression.hpp"
#include "Expression.hpp"
#include "types.h"

namespace PMacc
{
namespace lambda
{
namespace CT
{

template<typename Expr, typename CTExpr>
struct FillTerminalList;

template<typename Child0, typename CTExpr>
struct FillTerminalList<lambda::Expression<exprTypes::terminal, mpl::vector<Child0> >, CTExpr>
{
    typedef lambda::Expression<exprTypes::terminal, mpl::vector<Child0> > Expr;

    template<typename TerminalTuple>
    HDINLINE void operator()(const Expr& expr, TerminalTuple& terminalTuple) const
    {
        (Child0&)terminalTuple.at(mpl::int_<CTExpr::nextTerminalTypeIdx-1>()) = expr.child(mpl::int_<0>());
    }
};

template<int I, typename CTExpr>
struct FillTerminalList<lambda::Expression<exprTypes::terminal, mpl::vector<placeholder<I> > >, CTExpr>
{
    typedef lambda::Expression<exprTypes::terminal, mpl::vector<placeholder<I> > > Expr;

    template<typename TerminalTuple>
    HDINLINE void operator()(Expr, TerminalTuple) const {}
};

template<int I, typename CTExpr>
struct FillTerminalList<lambda::Expression<exprTypes::terminal, mpl::vector<mpl::int_<I> > >, CTExpr>
{
    typedef lambda::Expression<exprTypes::terminal, mpl::vector<mpl::int_<I> > > Expr;

    template<typename TerminalTuple>
    HDINLINE void operator()(Expr, TerminalTuple) const {}
};

template<typename Child0, typename Child1, typename CTExpr>
struct FillTerminalList<lambda::Expression<exprTypes::assign, mpl::vector<Child0, Child1> >, CTExpr>
{
    typedef lambda::Expression<exprTypes::assign, mpl::vector<Child0, Child1> > Expr;

    template<typename TerminalTuple>
    HDINLINE void operator()(const Expr& expr, TerminalTuple& terminalTuple) const
    {
        FillTerminalList<Child0, typename CTExpr::Child0>()(expr.child(mpl::int_<0>()), terminalTuple);
        FillTerminalList<Child1, typename CTExpr::Child1>()(expr.child(mpl::int_<1>()), terminalTuple);
    }
};

template<typename Child0, typename Child1, typename CTExpr>
struct FillTerminalList<lambda::Expression<exprTypes::plus, mpl::vector<Child0, Child1> >, CTExpr>
{
    typedef lambda::Expression<exprTypes::plus, mpl::vector<Child0, Child1> > Expr;

    template<typename TerminalTuple>
    HDINLINE void operator()(const Expr& expr, TerminalTuple& terminalTuple) const
    {
        FillTerminalList<Child0, typename CTExpr::Child0>()(expr.child(mpl::int_<0>()), terminalTuple);
        FillTerminalList<Child1, typename CTExpr::Child1>()(expr.child(mpl::int_<1>()), terminalTuple);
    }
};

template<typename Child0, typename Child1, typename CTExpr>
struct FillTerminalList<lambda::Expression<exprTypes::minus, mpl::vector<Child0, Child1> >, CTExpr>
{
    typedef lambda::Expression<exprTypes::minus, mpl::vector<Child0, Child1> > Expr;

    template<typename TerminalTuple>
    HDINLINE void operator()(const Expr& expr, TerminalTuple& terminalTuple) const
    {
        FillTerminalList<Child0, typename CTExpr::Child0>()(expr.child(mpl::int_<0>()), terminalTuple);
        FillTerminalList<Child1, typename CTExpr::Child1>()(expr.child(mpl::int_<1>()), terminalTuple);
    }
};

template<typename Child0, typename Child1, typename CTExpr>
struct FillTerminalList<lambda::Expression<exprTypes::multiply, mpl::vector<Child0, Child1> >, CTExpr>
{
    typedef lambda::Expression<exprTypes::multiply, mpl::vector<Child0, Child1> > Expr;

    template<typename TerminalTuple>
    HDINLINE void operator()(const Expr& expr, TerminalTuple& terminalTuple) const
    {
        FillTerminalList<Child0, typename CTExpr::Child0>()(expr.child(mpl::int_<0>()), terminalTuple);
        FillTerminalList<Child1, typename CTExpr::Child1>()(expr.child(mpl::int_<1>()), terminalTuple);
    }
};

template<typename Child0, typename Child1, typename CTExpr>
struct FillTerminalList<lambda::Expression<exprTypes::divide, mpl::vector<Child0, Child1> >, CTExpr>
{
    typedef lambda::Expression<exprTypes::divide, mpl::vector<Child0, Child1> > Expr;

    template<typename TerminalTuple>
    HDINLINE void operator()(const Expr& expr, TerminalTuple& terminalTuple) const
    {
        FillTerminalList<Child0, typename CTExpr::Child0>()(expr.child(mpl::int_<0>()), terminalTuple);
        FillTerminalList<Child1, typename CTExpr::Child1>()(expr.child(mpl::int_<1>()), terminalTuple);
    }
};

template<typename Child0, typename Child1, typename CTExpr>
struct FillTerminalList<lambda::Expression<exprTypes::comma, mpl::vector<Child0, Child1> >, CTExpr>
{
    typedef lambda::Expression<exprTypes::comma, mpl::vector<Child0, Child1> > Expr;

    template<typename TerminalTuple>
    HDINLINE void operator()(const Expr& expr, TerminalTuple& terminalTuple) const
    {
        FillTerminalList<Child0, typename CTExpr::Child0>()(expr.child(mpl::int_<0>()), terminalTuple);
        FillTerminalList<Child1, typename CTExpr::Child1>()(expr.child(mpl::int_<1>()), terminalTuple);
    }
};

#define FILLTERMINALLIST(Z,N,_) \
    FillTerminalList<Child ## N, typename CTExpr::Child ## N>()(expr.child(mpl::int_< N >()), terminalTuple);

#define OPERATOR_CALL(Z,N,_) \
    template<BOOST_PP_ENUM_PARAMS(N, typename Child), typename CTExpr> \
    struct FillTerminalList<lambda::Expression<exprTypes::call, mpl::vector<BOOST_PP_ENUM_PARAMS(N, Child)> >, CTExpr> \
    { \
        typedef lambda::Expression<exprTypes::call, mpl::vector<BOOST_PP_ENUM_PARAMS(N, Child)> > Expr; \
        \
        template<typename TerminalTuple> \
        HDINLINE void operator()(const Expr& expr, TerminalTuple& terminalTuple) const \
        { \
            BOOST_PP_REPEAT(N, FILLTERMINALLIST, _) \
        } \
    };

BOOST_PP_REPEAT_FROM_TO(2, LAMBDA_MAX_PARAMS, OPERATOR_CALL, _)

#undef FILLTERMINALLIST
#undef OPERATOR_CALL
/*
template<typename Child0, typename Child1, typename CTExpr>
struct FillTerminalList<lambda::Expression<exprTypes::call, mpl::vector<Child0, Child1> >, CTExpr>
{
    typedef lambda::Expression<exprTypes::call, mpl::vector<Child0, Child1> > Expr;

    template<typename TerminalTuple>
    HDINLINE void operator()(const Expr& expr, TerminalTuple& terminalTuple) const
    {
        FillTerminalList<Child0, typename CTExpr::Child0>()(expr.child(mpl::int_<0>()), terminalTuple);
        FillTerminalList<Child1, typename CTExpr::Child1>()(expr.child(mpl::int_<1>()), terminalTuple);
    }
};

template<typename Child0, typename Child1, typename Child2, typename CTExpr>
struct FillTerminalList<lambda::Expression<exprTypes::call, mpl::vector<Child0, Child1, Child2> >, CTExpr>
{
    typedef lambda::Expression<exprTypes::call, mpl::vector<Child0, Child1, Child2> > Expr;

    template<typename TerminalTuple>
    HDINLINE void operator()(const Expr& expr, TerminalTuple& terminalTuple) const
    {
        FillTerminalList<Child0, typename CTExpr::Child0>()(expr.child(mpl::int_<0>()), terminalTuple);
        FillTerminalList<Child1, typename CTExpr::Child1>()(expr.child(mpl::int_<1>()), terminalTuple);
        FillTerminalList<Child2, typename CTExpr::Child2>()(expr.child(mpl::int_<2>()), terminalTuple);
    }
};

template<typename Child0, typename Child1, typename Child2, typename Child3, typename CTExpr>
struct FillTerminalList<lambda::Expression<exprTypes::call, mpl::vector<Child0, Child1, Child2, Child3> >, CTExpr>
{
    typedef lambda::Expression<exprTypes::call, mpl::vector<Child0, Child1, Child2, Child3> > Expr;

    template<typename TerminalTuple>
    HDINLINE void operator()(const Expr& expr, TerminalTuple& terminalTuple) const
    {
        FillTerminalList<Child0, typename CTExpr::Child0>()(expr.child(mpl::int_<0>()), terminalTuple);
        FillTerminalList<Child1, typename CTExpr::Child1>()(expr.child(mpl::int_<1>()), terminalTuple);
        FillTerminalList<Child2, typename CTExpr::Child2>()(expr.child(mpl::int_<2>()), terminalTuple);
        FillTerminalList<Child3, typename CTExpr::Child3>()(expr.child(mpl::int_<3>()), terminalTuple);
    }
};
*/
template<typename Child0, typename Child1, typename CTExpr>
struct FillTerminalList<lambda::Expression<exprTypes::subscript, mpl::vector<Child0, Child1> >, CTExpr>
{
    typedef lambda::Expression<exprTypes::subscript, mpl::vector<Child0, Child1> > Expr;

    template<typename TerminalTuple>
    HDINLINE void operator()(const Expr& expr, TerminalTuple& terminalTuple) const
    {
        FillTerminalList<Child0, typename CTExpr::Child0>()(expr.child(mpl::int_<0>()), terminalTuple);
        FillTerminalList<Child1, typename CTExpr::Child1>()(expr.child(mpl::int_<1>()), terminalTuple);
    }
};

} // CT
} // lambda
} // PMacc

