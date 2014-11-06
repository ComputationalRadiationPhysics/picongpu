/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
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

#ifndef MATH_FUNCTOR_MIN_HPP
#define MATH_FUNCTOR_MIN_HPP


#include "types.h"
#include "lambda/Expression.hpp"

namespace PMacc
{
namespace math
{
namespace math_functor
{

struct Min
{
    template<typename Type>
    HDINLINE
    Type operator()(const Type& a, const Type& b) const
    {
        return min(a,b);
    }
};

lambda::Expression<lambda::exprTypes::terminal, mpl::vector<Min> > _min;

} // math_functor
} // math

namespace result_of
{

template<typename Type>
struct Functor<math::math_functor::Min, Type, Type>
{
    typedef Type type;
};

}

} // PMacc

#endif // MATH_FUNCTOR_MIN_HPP
