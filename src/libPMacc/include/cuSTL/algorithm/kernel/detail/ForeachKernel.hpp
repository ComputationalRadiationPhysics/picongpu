/**
 * Copyright 2013-2016 Heiko Burau
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

namespace PMacc
{
namespace algorithm
{
namespace kernel
{

#ifndef FOREACH_KERNEL_MAX_PARAMS
#define FOREACH_KERNEL_MAX_PARAMS 4
#endif

namespace detail
{

#define SHIFTACCESS_CURSOR(Z, N, _) forward(c ## N [cellIndex])

#define KERNEL_FOREACH(Z, N, _) \
/*                        typename C0, ..., typename CN     */ \
template<typename Mapper, BOOST_PP_ENUM_PARAMS(N, typename C), typename Functor> \
/*                                          C0 c0, ..., CN cN   */ \
DINLINE void operator()(Mapper mapper, BOOST_PP_ENUM_BINARY_PARAMS(N, C, c), Functor functor) const \
{ \
    math::Int<Mapper::dim> cellIndex(mapper(blockIdx, threadIdx)); \
/*          forward(c0[cellIndex]), ..., forward(cN[cellIndex])     */ \
    functor(BOOST_PP_ENUM(N, SHIFTACCESS_CURSOR, _)); \
}

struct KernelForeach
{
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(FOREACH_KERNEL_MAX_PARAMS), KERNEL_FOREACH, _)
};
#undef KERNEL_FOREACH
#undef SHIFTACCESS_CURSOR

} // namespace detail
} // namespace kernel
} // namespace algorithm
} // namespace PMacc
