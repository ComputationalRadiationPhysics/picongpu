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

#include "types.h"
#include "is_Expression.hpp"
#include "CT/TerminalTL.hpp"
#include "CT/Expression.hpp"
#include "CT/Eval.hpp"
#include "CT/FillTerminalList.hpp"
#include <math/Tuple.hpp>
#include "RefWrapper.hpp"
 #include <boost/type_traits/is_reference.hpp>

#include <boost/mpl/if.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/transform.hpp>

#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>

#ifndef LAMBDA_MAX_PARAMS
#define LAMBDA_MAX_PARAMS 8
#endif

namespace mpl = boost::mpl;

// forward declaration
namespace PMacc
{
namespace lambda
{
template<typename Expr>
struct ExprFunctor;
} // lambda
} // PMacc

namespace PMacc
{
namespace result_of
{

#define FUNCTOR(Z,N,_) \
    template<typename Expr, BOOST_PP_ENUM_PARAMS(N, typename Arg)> \
    struct Functor<lambda::ExprFunctor<Expr>, BOOST_PP_ENUM_PARAMS(N, Arg)> \
    { \
        typedef math::Tuple<mpl::vector<BOOST_PP_ENUM_PARAMS(N, Arg)> > ArgTuple; \
        typedef typename lambda::CT::result_of::Eval<Expr, ArgTuple>::type type; \
    };

BOOST_PP_REPEAT_FROM_TO(1, LAMBDA_MAX_PARAMS, FUNCTOR, _)
#undef FUNCTOR

} // result_of
} // PMacc

namespace PMacc
{
namespace lambda
{

template<typename Expr>
struct ExprFunctor
{
    typedef typename math::Tuple<
        typename CT::TerminalTL<Expr>::type> TerminalTuple;
    typedef CT::Expression<Expr, 0> CTExpr;

    TerminalTuple terminalTuple;

    HDINLINE
    ExprFunctor() {};

    HDINLINE
    ExprFunctor(const Expr& expr)
    {
        CT::FillTerminalList<Expr, CTExpr>()(expr, this->terminalTuple);
    }

    #define CREF_TYPE_LIST(Z, N, _) const Arg ## N &

    #define OPERATOR_CALL(Z,N,_)                                                                        \
        template<BOOST_PP_ENUM_PARAMS(N, typename Arg)>                                                 \
        HDINLINE                                                                                        \
        typename ::PMacc::result_of::Functor<ExprFunctor<Expr>, BOOST_PP_ENUM_PARAMS(N, Arg)>::type     \
        operator()(BOOST_PP_ENUM_BINARY_PARAMS(N, const Arg, &arg)) const                               \
        {                                                                                               \
            typedef mpl::vector<BOOST_PP_ENUM(N, CREF_TYPE_LIST, _)> ArgTypes;                             \
            typedef math::Tuple<ArgTypes> ArgTuple;                                                     \
            ArgTuple args(BOOST_PP_ENUM_PARAMS(N, arg));                                                \
            return CT::Eval<CTExpr>()(this->terminalTuple, args);                                       \
        }

    BOOST_PP_REPEAT_FROM_TO(1, LAMBDA_MAX_PARAMS, OPERATOR_CALL, _)

    #undef OPERATOR_CALL
    #undef CREF_TYPE_LIST
};

namespace result_of
{
template<typename T>
struct make_Functor
{
    typedef T type;
};

template<typename ExprType, typename Childs>
struct make_Functor<Expression<ExprType, Childs> >
{
    typedef ExprFunctor<Expression<ExprType, Childs> > type;
};
} // result_of

/** \return: if argument t is not an expression type, simply return t
 */
template<typename T>
HDINLINE
T make_Functor(const T& t)
{
    return t;
}

/** \return: if argument t is an expression type, return a functor made of t
 */
template<typename ExprType, typename Childs>
HDINLINE
ExprFunctor<Expression<ExprType, Childs> >
make_Functor(const Expression<ExprType, Childs>& expr)
{
    return ExprFunctor<Expression<ExprType, Childs> >(expr);
}

} // lambda
} // PMacc

