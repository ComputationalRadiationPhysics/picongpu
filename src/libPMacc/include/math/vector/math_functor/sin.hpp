/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera, Richard Pausch
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

#include "pmacc_types.hpp"
#include "lambda/Expression.hpp"
#include "algorithms/math/defines/trigo.hpp"

namespace PMacc
{
namespace math
{
namespace math_functor
{

template<typename T_Type>
struct Sin
{
    typedef T_Type result_type;

    DINLINE result_type operator()(const result_type& value) const
    {
        return algorithms::math::sin(value);
    }
};

lambda::Expression<lambda::exprTypes::terminal, mpl::vector<Sin<float>> > _sinf;
lambda::Expression<lambda::exprTypes::terminal, mpl::vector<Sin<double>> > _sind;

} // math_functor
} // math
} // PMacc

