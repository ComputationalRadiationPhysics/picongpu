/**
 * Copyright 2013-2014 Heiko Burau, Rene Widera
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

#pragma once

#include <stdint.h>
#include "math/Vector.hpp"
#include <boost/mpl/integral_c.hpp>
#include "traits/Limits.hpp"

namespace PMacc
{
namespace math
{
namespace CT
{

/** Compile time size_t vector
 *
 *
 * @tparam x value for x allowed range [0;max size_t value -1]
 * @tparam y value for y allowed range [0;max size_t value -1]
 * @tparam z value for z allowed range [0;max size_t value -1]
 *
 * default parameter is used to distinguish between values given by
 * the user and unset values.
 */
template<size_t x = traits::limits::Max<size_t>::value,
         size_t y = traits::limits::Max<size_t>::value,
         size_t z = traits::limits::Max<size_t>::value>
struct Size_t : public CT::Vector<mpl::integral_c<size_t, x>,
                              mpl::integral_c<size_t, y>,
                              mpl::integral_c<size_t, z> >
{};

template<>
struct Size_t<> : public CT::Vector<>
{};

template<size_t x>
struct Size_t<x> : public CT::Vector<mpl::integral_c<size_t, x> >
{};

template<size_t x, size_t y>
struct Size_t<x, y> : public CT::Vector<mpl::integral_c<size_t, x>,
                                    mpl::integral_c<size_t, y> >
{};

} // CT
} // math
} // PMacc
