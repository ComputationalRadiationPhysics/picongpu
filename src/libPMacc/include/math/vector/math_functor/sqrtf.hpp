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
 
#ifndef MATH_FUNCTOR_SQRTF_HPP
#define MATH_FUNCTOR_SQRTF_HPP

#include "types.h"
#include "lambda/Expression.hpp"

namespace PMacc
{
namespace math
{
namespace math_functor
{
    
struct Sqrtf
{
    typedef float result_type;
    
    HDINLINE float operator()(const float& value) const
    {
        return __sqrtf(value);
    }
};

lambda::Expression<lambda::exprTypes::terminal, mpl::vector<Sqrtf> > _sqrtf;
    
} // math_functor
} // math
} // PMacc

#endif // MATH_FUNCTOR_SQRTF_HPP
