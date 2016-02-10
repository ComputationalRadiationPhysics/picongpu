/**
 * Copyright 2013-2016 Rene Widera
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

namespace PMacc
{
namespace compileTime
{

namespace accessors
{

/** Get second type of the given type
 *
 * \tparam T type from which we return the second held type
 *
 * T must have defined ::second
 */
template<typename T>
struct Second
{
    typedef typename T::second type;
};

}//namespace accessors

}//namespace compileTime

}//namespace  PMacc
