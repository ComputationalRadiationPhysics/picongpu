/* Copyright 2013-2017 Heiko Burau
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

namespace pmacc
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

#define SHIFTACCESS_CURSOR(Z, N, _) c ## N [cellIndex]

#define KERNEL_FOREACH(Z, N, _) \
/*                        typename C0, ..., typename CN     */ \
template<typename Mapper, BOOST_PP_ENUM_PARAMS(N, typename C), typename Functor, typename T_Acc> \
/*                                          C0 c0, ..., CN cN   */ \
DINLINE void operator()(T_Acc const & acc, Mapper mapper, BOOST_PP_ENUM_BINARY_PARAMS(N, C, c), Functor functor) const \
{ \
    math::Int<Mapper::dim> cellIndex(mapper(acc, dim3(blockIdx), dim3(threadIdx))); \
/*          c0[cellIndex]), ..., cN[cellIndex]     */ \
    functor(acc, BOOST_PP_ENUM(N, SHIFTACCESS_CURSOR, _)); \
}

struct KernelForeach
{
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(FOREACH_KERNEL_MAX_PARAMS), KERNEL_FOREACH, _)
};
#undef KERNEL_FOREACH
#undef SHIFTACCESS_CURSOR

struct KernelForeachLockstep
{
    /** call functor
     *
     * Each argument is shifted to the origin of the block before it is passed
     * to the functor.
     */
    template<
        typename T_Acc,
        typename T_Mapper,
        typename T_Functor,
        typename... T_Args>
    ALPAKA_FN_ACC void operator()(
        T_Acc const & acc,
        T_Mapper const mapper,
        T_Functor functor,
        T_Args ... args
    ) const
    {
        // map to the origin of the block
        math::Int<
            T_Mapper::dim
        > cellIndex(
            mapper(
                acc,
                dim3( blockIdx ),
                dim3(
                    0,
                    0,
                    0
                )
            )
        );

        functor(
            acc,
            args[ cellIndex ]...
        );
    }
};

} // namespace detail
} // namespace kernel
} // namespace algorithm
} // namespace pmacc
