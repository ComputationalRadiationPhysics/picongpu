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

#include "is_Expression.hpp"
#include "ExprTypes.h"
#include <types.h>
#include <boost/mpl/vector.hpp>

namespace mpl = boost::mpl;

namespace PMacc
{
namespace lambda
{

template<typename _ExprType, typename _Childs>
struct Expression;

namespace result_of
{

template<typename T>
struct make_Expr
{
    typedef Expression<exprTypes::terminal, mpl::vector<T> > type;
};

template<typename ExprType, typename Childs>
struct make_Expr<Expression<ExprType, Childs> >
{
    typedef Expression<ExprType, Childs> type;
};

} // result_of

/** \return: returns a terminal expression made of argument t,
 * if t is not already an expression type
 */
template<typename T>
HDINLINE
Expression<exprTypes::terminal, mpl::vector<T> >
make_Expr(const T& t)
{
    return Expression<exprTypes::terminal, mpl::vector<T> >(t);
}

/** \return: if t is an expression type, simply return t
 */
template<typename ExprType, typename Childs>
HDINLINE
Expression<ExprType, Childs>
make_Expr(const Expression<ExprType, Childs>& expr)
{
    return expr;
}

// short version of make_Expr
template<typename T>
HDINLINE
typename result_of::make_Expr<T>::type
expr(const T& t)
{
    return make_Expr(t);
}

} // lambda
} // PMacc

