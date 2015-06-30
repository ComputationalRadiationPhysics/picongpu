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

#ifndef ALGORITHM_KERNEL_REDUCE_HPP
#define ALGORITHM_KERNEL_REDUCE_HPP

#include "math/vector/Int.hpp"
#include "cuSTL/container/CartBuffer.hpp"
#include "cuSTL/container/DeviceBuffer.hpp"
#include "cuSTL/zone/SphericZone.hpp"
#include <boost/mpl/void.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_same.hpp>

namespace mpl = boost::mpl;

namespace PMacc
{
namespace algorithm
{
namespace kernel
{

/** Reduce algorithm that calls a cuda kernel
 *
 */
struct Reduce
{

/* \param srcCursor Cursor located at the origin of the area of reduce
 * \param p_zone Zone of cells spanning the area of reduce
 * \param functor Functor with two arguments which returns the result of the reduce operation.
 *        Can also be a lambda expression.
 */
template<typename SrcCursor, typename Zone, typename Functor>
typename SrcCursor::ValueType operator()(const SrcCursor& srcCursor, const Zone& p_zone, const Functor& functor);
};

} // kernel
} // algorithm
} // PMacc

#include "Reduce.tpp"

#endif //ALGORITHM_KERNEL_REDUCE_HPP
