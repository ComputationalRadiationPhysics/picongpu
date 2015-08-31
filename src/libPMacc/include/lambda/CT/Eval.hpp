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

#define BOOST_BIND_NO_PLACEHOLDERS


#include "Expression.hpp"
#include "../placeholder.h"
#include "../ExprTypes.h"
#include "types.h"
#include <boost/mpl/void.hpp>
#include <boost/mpl/int.hpp>
#include <boost/type_traits/add_reference.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <math/Tuple.hpp>
#include "RefWrapper.hpp"

namespace mpl = boost::mpl;

namespace PMacc
{
namespace lambda
{
namespace CT
{

namespace result_of
{

namespace detail
{

template<typename Type>
struct RefWrapper2Ref
{
    typedef Type type;
};

template<typename Type>
struct RefWrapper2Ref<RefWrapper<Type> >
{
    typedef Type& type;
};

template<typename Type>
struct RefWrapper2Ref<const RefWrapper<Type> >
{
    typedef Type& type;
};

template<typename Type>
struct RefWrapper2Ref<const RefWrapper<Type>& >
{
    typedef Type& type;
};

}

template<typename Expr, typename ArgsTuple>
struct Eval;

template<typename Child0, typename ArgsTuple>
struct Eval<lambda::Expression<exprTypes::terminal, mpl::vector<Child0> >, ArgsTuple>
{
    typedef Child0 type;
};

template<typename Child0, typename ArgsTuple>
struct Eval<lambda::Expression<exprTypes::terminal, mpl::vector<RefWrapper<Child0> > >, ArgsTuple>
{
    typedef Child0& type;
};

template<typename Child0, typename ArgsTuple>
struct Eval<lambda::Expression<exprTypes::terminal, mpl::vector<const RefWrapper<Child0> > >, ArgsTuple>
{
    typedef Child0& type;
};

template<typename Child0, typename ArgsTuple>
struct Eval<lambda::Expression<exprTypes::terminal, mpl::vector<const RefWrapper<Child0>& > >, ArgsTuple>
{
    typedef Child0& type;
};

template<int I, typename ArgsTuple>
struct Eval<lambda::Expression<exprTypes::terminal, mpl::vector<placeholder<I> > >, ArgsTuple>
{
    typedef typename detail::RefWrapper2Ref<
        typename math::result_of::at_c<ArgsTuple, I>::type>::type type;
};

template<int I, typename ArgsTuple>
struct Eval<lambda::Expression<exprTypes::terminal, mpl::vector<mpl::int_<I> > >, ArgsTuple>
{
    typedef int type;
};

template<typename Child0, typename Child1, typename ArgsTuple>
struct Eval<lambda::Expression<exprTypes::assign, mpl::vector<Child0, Child1> >, ArgsTuple>
{
    typedef typename boost::add_reference<
        typename detail::RefWrapper2Ref<
        typename result_of::Eval<Child0, ArgsTuple>::type>::type>::type type;
};

template<typename Child0, typename Child1, typename ArgsTuple>
struct Eval<lambda::Expression<exprTypes::plus, mpl::vector<Child0, Child1> >, ArgsTuple>
{
    typedef typename boost::remove_reference<typename
        result_of::Eval<Child0, ArgsTuple>::type>::type type;
};

template<typename Child0, typename Child1, typename ArgsTuple>
struct Eval<lambda::Expression<exprTypes::minus, mpl::vector<Child0, Child1> >, ArgsTuple>
{
    typedef typename boost::remove_reference<typename
        result_of::Eval<Child0, ArgsTuple>::type>::type type;
};

template<typename Child0, typename Child1, typename ArgsTuple>
struct Eval<lambda::Expression<exprTypes::multiply, mpl::vector<Child0, Child1> >, ArgsTuple>
{
    typedef typename boost::remove_reference<typename
        result_of::Eval<Child0, ArgsTuple>::type>::type type;
};

template<typename Child0, typename Child1, typename ArgsTuple>
struct Eval<lambda::Expression<exprTypes::divide, mpl::vector<Child0, Child1> >, ArgsTuple>
{
    typedef typename boost::remove_reference<typename
        result_of::Eval<Child0, ArgsTuple>::type>::type type;
};

#define TD_ARG(Z,N,_) \
    typedef typename boost::remove_reference<typename \
            result_of::Eval<Child ## N, ArgsTuple>::type>::type BOOST_PP_CAT(Arg, BOOST_PP_DEC(N));

#define OPERATOR_CALL(Z,N,_) \
    template<BOOST_PP_ENUM_PARAMS(N, typename Child), typename ArgsTuple> \
    struct Eval<lambda::Expression<exprTypes::call, mpl::vector<BOOST_PP_ENUM_PARAMS(N, Child)> >, ArgsTuple> \
    { \
        typedef typename boost::remove_reference<typename \
            result_of::Eval<Child0, ArgsTuple>::type>::type _Functor; \
        \
        BOOST_PP_REPEAT_FROM_TO(1, N, TD_ARG, _) \
        \
        typedef typename ::PMacc::result_of::Functor<_Functor, BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), Arg)>::type type; \
    };

BOOST_PP_REPEAT_FROM_TO(2, LAMBDA_MAX_PARAMS, OPERATOR_CALL, _)

#undef TD_ARG
#undef OPERATOR_CALL

template<typename Child0, typename Child1, typename ArgsTuple>
struct Eval<lambda::Expression<exprTypes::subscript, mpl::vector<Child0, Child1> >, ArgsTuple>
{
    typedef typename boost::add_reference<
        typename boost::remove_reference<
        typename detail::RefWrapper2Ref<
        typename result_of::Eval<Child0, ArgsTuple>::type>::type>::type::type>::type type;
};

} // result_of

template<typename Expr>
struct Eval;

template<int I, int terminalTypeIdx>
struct Eval<CT::Expression<lambda::Expression<exprTypes::terminal, mpl::vector<placeholder<I> > >,
                           terminalTypeIdx> >
{
    typedef lambda::Expression<exprTypes::terminal, mpl::vector<placeholder<I> > > Expr;

    template<typename TerminalTuple, typename ArgsTuple>
    HDINLINE
    typename CT::result_of::Eval<Expr, ArgsTuple>::type
    operator()(TerminalTuple, const ArgsTuple& argsTuple) const
    {
        return argsTuple.at(mpl::int_<I>());
    }
};

template<int I, int terminalTypeIdx>
struct Eval<CT::Expression<lambda::Expression<exprTypes::terminal, mpl::vector<mpl::int_<I> > >,
                           terminalTypeIdx> >
{
    template<typename TerminalTuple, typename ArgsTuple>
    HDINLINE int
    operator()(TerminalTuple, ArgsTuple) const
    {
        return I;
    }
};

template<typename Child0, int terminalTypeIdx>
struct Eval<CT::Expression<lambda::Expression<exprTypes::terminal, mpl::vector<Child0> >,
                           terminalTypeIdx> >
{
    template<typename TerminalTuple, typename ArgsTuple>
    HDINLINE Child0
    operator()(const TerminalTuple& terminalTuple, ArgsTuple) const
    {
        return terminalTuple.at(mpl::int_<terminalTypeIdx>());
    }
};

template<typename Child0, int terminalTypeIdx>
struct Eval<CT::Expression<lambda::Expression<exprTypes::terminal, mpl::vector<RefWrapper<Child0> > >,
                           terminalTypeIdx> >
{
    template<typename TerminalTuple, typename ArgsTuple>
    HDINLINE Child0&
    operator()(const TerminalTuple& terminalTuple, ArgsTuple) const
    {
        return terminalTuple.at(mpl::int_<terminalTypeIdx>()).get();
    }
};

template<typename Child0, typename Child1, int terminalTypeIdx>
struct Eval<CT::Expression<lambda::Expression<exprTypes::assign, mpl::vector<Child0, Child1> >,
                           terminalTypeIdx> >
{
    typedef lambda::Expression<exprTypes::assign, mpl::vector<Child0, Child1> > Expr;
    typedef CT::Expression<Expr, terminalTypeIdx> CTExpr;

    template<typename TerminalTuple, typename ArgsTuple>
    HDINLINE typename result_of::Eval<Expr, ArgsTuple>::type
    operator()(const TerminalTuple& terminalTuple, const ArgsTuple& argsTuple) const
    {
        CT::Eval<typename CTExpr::Child0>()(terminalTuple, argsTuple)
         = CT::Eval<typename CTExpr::Child1>()(terminalTuple, argsTuple);

        return CT::Eval<typename CTExpr::Child0>()(terminalTuple, argsTuple);
    }
};

template<typename Child0, typename Child1, int terminalTypeIdx>
struct Eval<CT::Expression<lambda::Expression<exprTypes::plus, mpl::vector<Child0, Child1> >,
                           terminalTypeIdx> >
{
    typedef lambda::Expression<exprTypes::plus, mpl::vector<Child0, Child1> > Expr;
    typedef CT::Expression<Expr, terminalTypeIdx> CTExpr;

    template<typename TerminalTuple, typename ArgsTuple>
    HDINLINE typename result_of::Eval<Expr, ArgsTuple>::type
    operator()(const TerminalTuple& terminalTuple, const ArgsTuple& argsTuple) const
    {
        return CT::Eval<typename CTExpr::Child0>()(terminalTuple, argsTuple)
         + CT::Eval<typename CTExpr::Child1>()(terminalTuple, argsTuple);
    }
};

template<typename Child0, typename Child1, int terminalTypeIdx>
struct Eval<CT::Expression<lambda::Expression<exprTypes::minus, mpl::vector<Child0, Child1> >,
                           terminalTypeIdx> >
{
    typedef lambda::Expression<exprTypes::minus, mpl::vector<Child0, Child1> > Expr;
    typedef CT::Expression<Expr, terminalTypeIdx> CTExpr;

    template<typename TerminalTuple, typename ArgsTuple>
    HDINLINE typename result_of::Eval<Expr, ArgsTuple>::type
    operator()(const TerminalTuple& terminalTuple, const ArgsTuple& argsTuple) const
    {
        return CT::Eval<typename CTExpr::Child0>()(terminalTuple, argsTuple)
         - CT::Eval<typename CTExpr::Child1>()(terminalTuple, argsTuple);
    }
};

template<typename Child0, typename Child1, int terminalTypeIdx>
struct Eval<CT::Expression<lambda::Expression<exprTypes::multiply, mpl::vector<Child0, Child1> >,
                           terminalTypeIdx> >
{
    typedef lambda::Expression<exprTypes::multiply, mpl::vector<Child0, Child1> > Expr;
    typedef CT::Expression<Expr, terminalTypeIdx> CTExpr;

    template<typename TerminalTuple, typename ArgsTuple>
    HDINLINE typename result_of::Eval<Expr, ArgsTuple>::type
    operator()(const TerminalTuple& terminalTuple, const ArgsTuple& argsTuple) const
    {
        return CT::Eval<typename CTExpr::Child0>()(terminalTuple, argsTuple)
         * CT::Eval<typename CTExpr::Child1>()(terminalTuple, argsTuple);
    }
};

template<typename Child0, typename Child1, int terminalTypeIdx>
struct Eval<CT::Expression<lambda::Expression<exprTypes::divide, mpl::vector<Child0, Child1> >,
                           terminalTypeIdx> >
{
    typedef lambda::Expression<exprTypes::divide, mpl::vector<Child0, Child1> > Expr;
    typedef CT::Expression<Expr, terminalTypeIdx> CTExpr;

    template<typename TerminalTuple, typename ArgsTuple>
    HDINLINE typename result_of::Eval<Expr, ArgsTuple>::type
    operator()(const TerminalTuple& terminalTuple, const ArgsTuple& argsTuple) const
    {
        return CT::Eval<typename CTExpr::Child0>()(terminalTuple, argsTuple)
         / CT::Eval<typename CTExpr::Child1>()(terminalTuple, argsTuple);
    }
};

template<typename Child0, typename Child1, int terminalTypeIdx>
struct Eval<CT::Expression<lambda::Expression<exprTypes::comma, mpl::vector<Child0, Child1> >,
                           terminalTypeIdx> >
{
    typedef CT::Expression<lambda::Expression<exprTypes::comma, mpl::vector<Child0, Child1> >,
                           terminalTypeIdx> CTExpr;
    typedef void result_type;

    template<typename TerminalTuple, typename ArgsTuple>
    HDINLINE void
    operator()(const TerminalTuple& terminalTuple, const ArgsTuple& argsTuple) const
    {
        CT::Eval<typename CTExpr::Child0>()(terminalTuple, argsTuple);
        CT::Eval<typename CTExpr::Child1>()(terminalTuple, argsTuple);
    }
};

#define EVAL_CHILD(Z,N,_) \
    CT::Eval<BOOST_PP_CAT(typename CTExpr::Child, BOOST_PP_INC(N))>()(terminalTuple, argsTuple)

#define OPERATOR_CALL(Z,N,_) \
    template<BOOST_PP_ENUM_PARAMS(N, typename Child), int terminalTypeIdx> \
    struct Eval<CT::Expression<lambda::Expression<exprTypes::call, mpl::vector<BOOST_PP_ENUM_PARAMS(N, Child)> >, \
                                terminalTypeIdx> > \
    { \
        typedef lambda::Expression<exprTypes::call, mpl::vector<BOOST_PP_ENUM_PARAMS(N, Child)> > Expr; \
        typedef CT::Expression<Expr, terminalTypeIdx> CTExpr; \
        \
        template<typename TerminalTuple, typename ArgsTuple> \
        HDINLINE typename result_of::Eval<Expr, ArgsTuple>::type \
        operator()(const TerminalTuple& terminalTuple, const ArgsTuple& argsTuple) const \
        { \
            return CT::Eval<typename CTExpr::Child0>()(terminalTuple, argsTuple)( \
                    BOOST_PP_ENUM(BOOST_PP_DEC(N), EVAL_CHILD, _)); \
        } \
    };

BOOST_PP_REPEAT_FROM_TO(2, LAMBDA_MAX_PARAMS, OPERATOR_CALL, _)
#undef EVAL_CHILD
#undef OPERATOR_CALL

template<typename Child0, typename Child1, int terminalTypeIdx>
struct Eval<CT::Expression<lambda::Expression<exprTypes::subscript, mpl::vector<Child0, Child1> >,
                           terminalTypeIdx> >
{
    typedef lambda::Expression<exprTypes::subscript, mpl::vector<Child0, Child1> > Expr;
    typedef CT::Expression<Expr, terminalTypeIdx> CTExpr;

    template<typename TerminalTuple, typename ArgsTuple>
    HDINLINE typename result_of::Eval<Expr, ArgsTuple>::type
    operator()(const TerminalTuple& terminalTuple, const ArgsTuple& argsTuple) const
    {
        return CT::Eval<typename CTExpr::Child0>()(terminalTuple, argsTuple)[
                CT::Eval<typename CTExpr::Child1>()(terminalTuple, argsTuple)];
    }
};

} // CT
} // lambda
} // PMacc

