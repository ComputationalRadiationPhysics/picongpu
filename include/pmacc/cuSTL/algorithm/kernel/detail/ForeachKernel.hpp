/* Copyright 2013-2021 Heiko Burau
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/dimensions/DataSpaceOperations.hpp"

#include <boost/preprocessor.hpp>


namespace pmacc
{
    namespace algorithm
    {
        namespace kernel
        {
#ifndef FOREACH_KERNEL_MAX_PARAMS
#    define FOREACH_KERNEL_MAX_PARAMS 4
#endif

            namespace detail
            {
#define SHIFTACCESS_CURSOR(Z, N, _) c##N[cellIndex]

#define KERNEL_FOREACH(Z, N, _)                                                                                       \
    /*                        typename C0, ..., typename CN     */                                                    \
    template<                                                                                                         \
        typename Mapper,                                                                                              \
        BOOST_PP_ENUM_PARAMS(N, typename C),                                                                          \
        typename Functor,                                                                                             \
        typename T_Acc> /*                                          C0 c0, ..., CN cN   */                            \
    DINLINE void operator()(T_Acc const& acc, Mapper mapper, BOOST_PP_ENUM_BINARY_PARAMS(N, C, c), Functor functor)   \
        const                                                                                                         \
    {                                                                                                                 \
        math::Int<Mapper::dim> cellIndex(                                                                             \
            mapper(acc, cupla::dim3(cupla::blockIdx(acc)), cupla::dim3(cupla::threadIdx(acc))));                      \
        /*          c0[cellIndex]), ..., cN[cellIndex]     */                                                         \
        functor(acc, BOOST_PP_ENUM(N, SHIFTACCESS_CURSOR, _));                                                        \
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
                    template<typename T_Acc, typename T_Mapper, typename T_Functor, typename... T_Args>
                    ALPAKA_FN_ACC void operator()(
                        T_Acc const& acc,
                        T_Mapper const mapper,
                        T_Functor functor,
                        T_Args... args) const
                    {
                        // map to the origin of the block
                        math::Int<T_Mapper::dim> cellIndex(
                            mapper(acc, cupla::dim3(cupla::blockIdx(acc)), cupla::dim3(0, 0, 0)));

                        functor(acc, args[cellIndex]...);
                    }
                };

                namespace RT
                {
                    /** Run a cuSTL KernelForeach
                     *
                     * Allow to run the cuSTL foreach with runtime block sizes.
                     * @warning collective functors which contain synchronization are not supported
                     */
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
                            typename T_BlockSize,
                            typename T_Functor,
                            typename... T_Args>
                        ALPAKA_FN_ACC void operator()(
                            T_Acc const& acc,
                            T_Mapper const mapper,
                            T_BlockSize const blockSize,
                            T_Functor functor,
                            T_Args... args) const
                        {
                            /* KernelForeachLockstep is always called as kernel with three dimensions
                             * therefore we have to reduce the dimension if the mapper is only 2D or 1D.
                             */
                            auto const blockSizeShrinked = blockSize.template shrink<T_Mapper::dim>();
                            uint32_t const domainElementCount = blockSizeShrinked.productOfComponents();
                            DataSpace<T_Mapper::dim> const domainSize(blockSizeShrinked);

                            // map to the origin of the block
                            math::Int<T_Mapper::dim> blockCellOffset(mapper(
                                acc,
                                domainSize.toDim3(),
                                cupla::dim3(cupla::blockIdx(acc)),
                                cupla::dim3(0, 0, 0)));


                            for(uint32_t i = cupla::threadIdx(acc).x; i < domainElementCount;
                                i += cupla::blockDim(acc).x)
                            {
                                auto const inBlockOffset = DataSpaceOperations<T_Mapper::dim>::map(domainSize, i);
                                auto const cellOffset = blockCellOffset + inBlockOffset;
                                functor(acc, args[cellOffset]...);
                            }
                        }
                    };
                } // namespace RT
            } // namespace detail
        } // namespace kernel
    } // namespace algorithm
} // namespace pmacc
