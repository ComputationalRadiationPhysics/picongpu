/* Copyright 2013-2021 Heiko Burau, Rene Widera
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

#include "pmacc/types.hpp"
#include "pmacc/math/vector/Size_t.hpp"
#include "pmacc/math/vector/Int.hpp"
#include "detail/SphericMapper.hpp"

#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include "pmacc/eventSystem/tasks/TaskKernel.hpp"
#include "pmacc/eventSystem/events/kernelEvents.hpp"

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
    /* typename C0, typename C1, ... */                                                                               \
    template<                                                                                                         \
        typename Mapper,                                                                                              \
        BOOST_PP_ENUM_PARAMS(N, typename C),                                                                          \
        typename Functor,                                                                                             \
        typename T_Acc> /* C0 c0, C1 c1, ... */                                                                       \
    DINLINE void operator()(T_Acc const& acc, Mapper mapper, BOOST_PP_ENUM_BINARY_PARAMS(N, C, c), Functor functor)   \
        const                                                                                                         \
    {                                                                                                                 \
        math::Int<Mapper::dim> cellIndex(mapper(acc, cupla::dim3(cupla::blockIdx(acc))));                             \
        /* c0[cellIndex], c1[cellIndex], ... */                                                                       \
        functor(acc, BOOST_PP_ENUM(N, SHIFTACCESS_CURSOR, _));                                                        \
    }

                struct KernelForeachBlock
                {
                    BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(FOREACH_KERNEL_MAX_PARAMS), KERNEL_FOREACH, _)
                };
#undef KERNEL_FOREACH
#undef SHIFTACCESS_CURSOR

            } // namespace detail

#define SHIFT_CURSOR_ZONE(Z, N, _) C##N c##N##_shifted = c##N(p_zone.offset);
#define SHIFTED_CURSOR(Z, N, _) c##N##_shifted

#define FOREACH_OPERATOR(Z, N, _)                                                                                     \
    /* typename C0, typename C1, ... */                                                                               \
    template<typename Zone, BOOST_PP_ENUM_PARAMS(N, typename C), typename Functor> /* C0 c0, C1 c1, ... */            \
    void operator()(const Zone& p_zone, BOOST_PP_ENUM_BINARY_PARAMS(N, C, c), const Functor& functor)                 \
    {                                                                                                                 \
        /* C0 c0_shifted = c0(p_zone.offset); */                                                                      \
        /* C1 c1_shifted = c1(p_zone.offset); */                                                                      \
        /* ... */                                                                                                     \
        BOOST_PP_REPEAT(N, SHIFT_CURSOR_ZONE, _)                                                                      \
                                                                                                                      \
        auto blockDim = ThreadBlock::toRT();                                                                          \
        detail::SphericMapper<Zone::dim, BlockDim> mapper;                                                            \
        using namespace pmacc;                                                                                        \
        PMACC_KERNEL(detail::KernelForeachBlock{})                                                                    \
        (mapper.cuplaGridDim(p_zone.size), blockDim) /* c0_shifted, c1_shifted, ... */                                \
            (mapper, BOOST_PP_ENUM(N, SHIFTED_CURSOR, _), functor);                                                   \
    }

            /** Special foreach algorithm that calls a cupla kernel
             *
             * Behaves like kernel::Foreach, except that is doesn't shift the cursors cell by cell, but
             * shifts them to the top left (front) corner cell of their corresponding cupla block.
             * So if BlockDim is 4x4x4 it shifts 64 cursors to (0,0,0), 64 to (4,0,0), 64 to (8,0,0), ...
             *
             * \tparam BlockDim 3D compile-time vector (pmacc::math::CT::Int) of the size of the cupla blockDim.
             * \tparam ThreadBlock ignored
             */
            template<typename BlockDim, typename ThreadBlock = BlockDim>
            struct ForeachBlock
            {
                /* operator()(zone, cursor0, cursor1, ..., cursorN-1, functor or lambdaFun)
                 *
                 * \param zone Accepts currently only a zone::SphericZone object (e.g. containerObj.zone())
                 * \param cursorN cursor for the N-th data source (e.g. containerObj.origin())
                 * \param functor or lambdaFun either a functor with N arguments or a N-ary lambda function (e.g. _1 =
                 * _2)
                 *
                 * It is called like functor(*cursor0(cellId), ..., *cursorN(cellId))
                 *
                 */
                BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(FOREACH_KERNEL_MAX_PARAMS), FOREACH_OPERATOR, _)
            };

#undef FOREACH_OPERATOR
#undef SHIFT_CURSOR_ZONE
#undef SHIFTED_CURSOR

        } // namespace kernel
    } // namespace algorithm
} // namespace pmacc
