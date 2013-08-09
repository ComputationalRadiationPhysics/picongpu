/**
 * Copyright 2013 Heiko Burau, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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
 
#ifndef LAMBDA_EXPRESSION_HPP
#define LAMBDA_EXPRESSION_HPP

#include "types.h"
#include "ExprTypes.h"
#include "placeholder.h"
#include "is_Expression.hpp"
#include "make_Expr.hpp"
#include <boost/mpl/void.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/int.hpp>
#include "make_Expr.hpp"

#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/cat.hpp>

#ifndef LAMBDA_MAX_PARAMS
#define LAMBDA_MAX_PARAMS 8
#endif

namespace mpl = boost::mpl;

namespace PMacc
{
namespace lambda
{
using mpl::at_c;
    
template<typename T, typename Dummy1, typename Dummy2>
struct BaseWrapper : public T 
{
HDINLINE BaseWrapper(const T& t) : T(t) {}
};

#define BASEWRAPPER_ARG(Z, N, _) BaseWrapper<typename at_c<_Childs, N >::type, _Childs, mpl::int_< N > >
#define BASEWRAPPER_TD(Z, N, _) typedef BaseWrapper<typename at_c<_Childs, N >::type, _Childs, mpl::int_< N > > BaseChild ## N;
#define CHILD_CARG(Z, N, _) const typename at_c<Childs, N >::type& child ## N = typename at_c<Childs, N >::type()
#define BASECHILD_CARG(Z, N, _) BaseChild ## N (child ## N)
#define GETCHILD(Z, N, _) HDINLINE typename at_c<Childs, N>::type getChild ## N() const {return (BaseChild ## N)(*this);}

template<typename _ExprType, typename _Childs>
struct ExpressionBase : public BOOST_PP_ENUM(LAMBDA_MAX_PARAMS, BASEWRAPPER_ARG, _)
{
    typedef ExpressionBase<_ExprType, _Childs> This;
    typedef _Childs Childs;
    BOOST_PP_REPEAT(LAMBDA_MAX_PARAMS, BASEWRAPPER_TD, _)
    
    HDINLINE
    ExpressionBase(BOOST_PP_ENUM(LAMBDA_MAX_PARAMS, CHILD_CARG, _)) 
     : BOOST_PP_ENUM(LAMBDA_MAX_PARAMS, BASECHILD_CARG, _) {}
    
    BOOST_PP_REPEAT(LAMBDA_MAX_PARAMS, GETCHILD, _)
};

#undef BASEWRAPPER_ARG
#undef BASEWRAPPER_TD
#undef CHILD_CARG
#undef BASECHILD_CARG
#undef GETCHILD

template<typename _Child0>
struct ExpressionBase<exprTypes::terminal, mpl::vector<_Child0> >
{
    typedef ExpressionBase<exprTypes::terminal, mpl::vector<_Child0> > This;
    typedef _Child0 Child0;
    
    Child0 child0;
    
    HDINLINE
    ExpressionBase(const _Child0& child0)
     : child0(child0) {}
    
    HDINLINE Child0 getChild0() const {return child0;}
};

#define CHILD_CARG(Z,N,_) const typename at_c<_Childs,N>::type& child ## N

#define EXPRESSION_CTOR(Z, N, _) \
    HDINLINE Expression(BOOST_PP_ENUM(N, CHILD_CARG, _)) \
     : Base(BOOST_PP_ENUM_PARAMS(N, child)) {}
    
template<typename _ExprType, typename _Childs>
struct Expression : public ExpressionBase<_ExprType, _Childs>
{
    typedef Expression<_ExprType, _Childs> This;
    typedef ExpressionBase<_ExprType, _Childs> Base;
    typedef _ExprType ExprType;
    
    HDINLINE Expression(const typename at_c<_Childs,0>::type& child0 = typename at_c<_Childs,0>::type())
     : Base(child0) {}
    
    BOOST_PP_REPEAT_FROM_TO(2, LAMBDA_MAX_PARAMS, EXPRESSION_CTOR, _)
    
    template<typename Rhs>
    HDINLINE
    Expression<exprTypes::assign, mpl::vector<This, typename result_of::make_Expr<Rhs>::type> > operator=(const Rhs& rhs) const
    {
        return Expression<exprTypes::assign, mpl::vector<This, typename result_of::make_Expr<Rhs>::type> >
        (*this, make_Expr(rhs));
    }
    
    template<typename Rhs>
    HDINLINE
    Expression<exprTypes::plus, mpl::vector<This, typename result_of::make_Expr<Rhs>::type> > operator+(const Rhs& rhs) const
    {
        return Expression<exprTypes::plus, mpl::vector<This, typename result_of::make_Expr<Rhs>::type> >
        (*this, make_Expr(rhs));
    }
    
    template<typename Rhs>
    HDINLINE
    Expression<exprTypes::minus, mpl::vector<This, typename result_of::make_Expr<Rhs>::type> > operator-(const Rhs& rhs) const
    {
        return Expression<exprTypes::minus, mpl::vector<This, typename result_of::make_Expr<Rhs>::type> >
        (*this, make_Expr(rhs));
    }
    
    template<typename Rhs>
    HDINLINE
    Expression<exprTypes::multiply, mpl::vector<This, typename result_of::make_Expr<Rhs>::type> > operator*(const Rhs& rhs) const
    {
        return Expression<exprTypes::multiply, mpl::vector<This, typename result_of::make_Expr<Rhs>::type> >
        (*this, make_Expr(rhs));
    }
    
    template<typename Rhs>
    HDINLINE
    Expression<exprTypes::divide, mpl::vector<This, typename result_of::make_Expr<Rhs>::type> > operator/(const Rhs& rhs) const
    {
        return Expression<exprTypes::divide, mpl::vector<This, typename result_of::make_Expr<Rhs>::type> >
        (*this, make_Expr(rhs));
    }
    
    template<typename Rhs>
    HDINLINE
    Expression<exprTypes::comma, mpl::vector<This, typename result_of::make_Expr<Rhs>::type> > operator,(const Rhs& rhs) const
    {
        return Expression<exprTypes::comma, mpl::vector<This, typename result_of::make_Expr<Rhs>::type> >
        (*this, make_Expr(rhs));
    }
    
    #define RESULT_OF_MAKE_EXPR(Z, N, _) typename result_of::make_Expr<Arg ## N>::type
    #define MAKE_EXPR(Z, N, _) make_Expr(arg ## N)
    
    #define OPERATOR_CALL(Z, N, _) \
        template<BOOST_PP_ENUM_PARAMS(N, typename Arg)> \
        HDINLINE \
        Expression<exprTypes::call, mpl::vector<This BOOST_PP_ENUM_TRAILING(N, RESULT_OF_MAKE_EXPR, _)> > \
        operator()(BOOST_PP_ENUM_BINARY_PARAMS(N, const Arg, &arg)) const \
        { \
            return Expression<exprTypes::call, mpl::vector<This BOOST_PP_ENUM_TRAILING(N, RESULT_OF_MAKE_EXPR, _)> > \
            (*this BOOST_PP_ENUM_TRAILING(N, MAKE_EXPR, _)); \
        }
    
    BOOST_PP_REPEAT_FROM_TO(1, LAMBDA_MAX_PARAMS, OPERATOR_CALL, _)
    
    #undef RESULT_OF_FUNCTOR_HPP
    #undef MAKE_EXPR
    #undef OPERATOR_CALL
    
    template<typename Arg>
    HDINLINE
    Expression<exprTypes::subscript, mpl::vector<This, typename result_of::make_Expr<Arg>::type> >
    operator[](const Arg& arg) const
    {
        return Expression<exprTypes::subscript, mpl::vector<This, typename result_of::make_Expr<Arg>::type> >
        (*this, make_Expr(arg));
    }
};

template<typename _ExprType, typename _Childs>
struct is_Expression<Expression<_ExprType, _Childs> >
{
    typedef mpl::bool_<true> type;
};

#define DECLARE_PLACEHOLDERS() \
const Expression<exprTypes::terminal, mpl::vector<placeholder<0> > > _1; \
const Expression<exprTypes::terminal, mpl::vector<placeholder<1> > > _2; \
const Expression<exprTypes::terminal, mpl::vector<placeholder<2> > > _3; \
const Expression<exprTypes::terminal, mpl::vector<placeholder<3> > > _4; \
const Expression<exprTypes::terminal, mpl::vector<placeholder<4> > > _5; \
const Expression<exprTypes::terminal, mpl::vector<placeholder<5> > > _6; \
const Expression<exprTypes::terminal, mpl::vector<placeholder<6> > > _7;

DECLARE_PLACEHOLDERS()

} // lambda
} // PMacc

#endif // LAMBDA_EXPRESSION_HPP
