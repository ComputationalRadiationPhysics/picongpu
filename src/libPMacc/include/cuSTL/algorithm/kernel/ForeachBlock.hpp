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
 
#ifndef ALGORITHM_KERNEL_FOREACHBLOCK_HPP
#define ALGORITHM_KERNEL_FOREACHBLOCK_HPP

#include "types.h"
#include "math/vector/Size_t.hpp"
#include "math/vector/Int.hpp"
#include "lambda/make_Functor.hpp"
#include "detail/SphericMapper.hpp"
#include "forward.hpp"

#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include "eventSystem/tasks/TaskKernel.hpp"
#include "eventSystem/events/kernelEvents.hpp"

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

#define KERNEL_FOREACH(Z, N, _)                                                                             \
                        /* typename C0, typename C1, ... */                                                 \
template<typename Mapper, BOOST_PP_ENUM_PARAMS(N, typename C), typename Functor>                            \
                                                /* C0 c0, C1 c1, ... */                                     \
__global__ void kernelForeachBlock(Mapper mapper, BOOST_PP_ENUM_BINARY_PARAMS(N, C, c), Functor functor)    \
{                                                                                                           \
    math::Int<Mapper::dim> cellIndex(mapper(blockIdx));                                                     \
         /* c0[cellIndex], c1[cellIndex], ... */                                                            \
    functor(BOOST_PP_ENUM(N, SHIFTACCESS_CURSOR, _));                                                       \
}

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(FOREACH_KERNEL_MAX_PARAMS), KERNEL_FOREACH, _)

#undef KERNEL_FOREACH
#undef SHIFTACCESS_CURSOR

}

#define SHIFT_CURSOR_ZONE(Z, N, _) C ## N c ## N ## _shifted = c ## N (_zone.offset);
#define SHIFTED_CURSOR(Z, N, _) c ## N ## _shifted

#define FOREACH_OPERATOR(Z, N, _)                                                                           \
    /* calls functor for each cell within zone with the block wise jumped and dereferenced cursors          \
     * \param _zone The zone                                                                                \
     * \param c0, c1, ... The cursors                                                                       \
     * \param functor The functor                                                                           \
     */                                                                                                     \
                         /* typename C0, typename C1, ... */                                                \
    template<typename Zone, BOOST_PP_ENUM_PARAMS(N, typename C), typename Functor>                          \
                                     /* C0 c0, C1 c1, ... */                                                \
    void operator()(const Zone& _zone, BOOST_PP_ENUM_BINARY_PARAMS(N, C, c), const Functor& functor)        \
    {                                                                                                       \
        /* C0 c0_shifted = c0(_zone.offset); */                                                             \
        /* C1 c1_shifted = c1(_zone.offset); */                                                             \
        /* ... */                                                                                           \
        BOOST_PP_REPEAT(N, SHIFT_CURSOR_ZONE, _)                                                            \
                                                                                                            \
        dim3 blockDim(ThreadBlock::x::value, ThreadBlock::y::value, ThreadBlock::z::value);                 \
        detail::SphericMapper<Zone::dim, BlockDim> mapper; \
        using namespace PMacc;                                                                              \
        __cudaKernel(detail::kernelForeachBlock)(mapper.cudaGridDim(_zone.size), blockDim)                  \
                    /* c0_shifted, c1_shifted, ... */                                                       \
            (mapper, BOOST_PP_ENUM(N, SHIFTED_CURSOR, _), lambda::make_Functor(functor));                   \
    }
    
/** Special foreach algorithm that calls a cuda kernel
 * 
 * Behaves like kernel::Foreach, except that is doesn't shift the cursors cell by cell, but
 * shifts them to the top left (front) corner cell of their corresponding cuda block.
 * So if BlockDim is 4x4x4 it shifts 64 cursors to (0,0,0), 64 to (4,0,0), 64 to (8,0,0), ...
 * 
 * \tparam BlockDim cuda block-dim size. Has to be a denominator of _zone.size
 * \tparam ThreadBlock The same as BlockDim as you see
 */
template<typename BlockDim, typename ThreadBlock = BlockDim>
struct ForeachBlock
{
    BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(FOREACH_KERNEL_MAX_PARAMS), FOREACH_OPERATOR, _)
};

#undef FOREACH_OPERATOR
#undef SHIFT_CURSOR_ZONE
#undef SHIFTED_CURSOR
    
} // kernel
} // algorithm
} // PMacc

#endif // ALGORITHM_KERNEL_FOREACHBLOCK_HPP
