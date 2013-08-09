/**
 * Copyright 2013 Heiko Burau, René Widera
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
    
#define SHIFTACCESS_CURSOR(Z, N, _) c ## N [cellIndex]    

#define KERNEL_FOREACH(Z, N, _) \
template<typename Mapper, BOOST_PP_ENUM_PARAMS(N, typename C), typename Functor> \
__global__ void kernelForeachBlock(Mapper mapper, BOOST_PP_ENUM_BINARY_PARAMS(N, C, c), Functor functor) \
{ \
    math::Int<Mapper::dim> cellIndex(mapper(blockIdx)); \
    functor(BOOST_PP_ENUM(N, SHIFTACCESS_CURSOR, _)); \
}

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(FOREACH_KERNEL_MAX_PARAMS), KERNEL_FOREACH, _)

#undef KERNEL_FOREACH
#undef SHIFTACCESS_CURSOR

}

#define SHIFT_CURSOR_ZONE(Z, N, _) C ## N c ## N ## _shifted = c ## N (_zone.offset);
#define SHIFTED_CURSOR(Z, N, _) c ## N ## _shifted

#define FOREACH_OPERATOR(Z, N, _) \
    template<typename Zone, BOOST_PP_ENUM_PARAMS(N, typename C), typename Functor> \
    void operator()(const Zone& _zone, BOOST_PP_ENUM_BINARY_PARAMS(N, C, c), const Functor& functor) \
    { \
        BOOST_PP_REPEAT(N, SHIFT_CURSOR_ZONE, _) \
        \
        dim3 blockDim(ThreadBlock::x::value, ThreadBlock::y::value, ThreadBlock::z::value); \
        detail::SphericMapper<Zone::dim, BlockDim> mapper(_zone.size); \
        using namespace PMacc; \
        __cudaKernel(detail::kernelForeachBlock)(mapper.cudaGridDim(_zone.size), blockDim) \
            (mapper, BOOST_PP_ENUM(N, SHIFTED_CURSOR, _), lambda::make_Functor(functor)); \
    }
    
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
