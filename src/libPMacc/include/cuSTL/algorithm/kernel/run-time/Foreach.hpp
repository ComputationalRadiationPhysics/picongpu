/**
 * Copyright 2013-2015 Heiko Burau, Rene Widera, Alexander Grund
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


#include "types.h"
#include "math/vector/Size_t.hpp"
#include "math/vector/Int.hpp"
#include "lambda/make_Functor.hpp"
#include "cuSTL/algorithm/kernel/detail/SphericMapper.hpp"
#include "cuSTL/algorithm/kernel/detail/ForeachKernel.hpp"
#include <forward.hpp>

#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include "eventSystem/tasks/TaskKernel.hpp"
#include "eventSystem/events/kernelEvents.hpp"
#include "Environment.hpp"
#include <cassert>

namespace PMacc
{
namespace algorithm
{
namespace kernel
{
namespace RT
{

#ifndef FOREACH_KERNEL_MAX_PARAMS
#define FOREACH_KERNEL_MAX_PARAMS 4
#endif

#define SHIFT_CURSOR_ZONE(Z, N, _) C ## N c ## N ## _shifted = c ## N (p_zone.offset);
#define SHIFTED_CURSOR(Z, N, _) c ## N ## _shifted

#define FOREACH_OPERATOR(Z, N, _)                                                                                   \
    /*                      typename C0, ..., typename CN            */                                             \
    template<typename Zone, BOOST_PP_ENUM_PARAMS(N, typename C), typename Functor>                                  \
                                /*     C0 c0, ..., CN cN  */                                                        \
    void operator()(const Zone& p_zone, BOOST_PP_ENUM_BINARY_PARAMS(N, C, c), const Functor& functor)                \
    {                                                                                                               \
        /* C0 c0_shifted = c0(p_zone.offset); ...; CN cN_shifted = cN(p_zone.offset); */                              \
        BOOST_PP_REPEAT(N, SHIFT_CURSOR_ZONE, _)                                                                    \
                                                                                                                    \
        /* the maximum number of threads per block for devices with                                                 \
         * compute capability > 2.0 is 1024 */                                                                      \
        assert(this->_blockDim.productOfComponents()<=1024);                                                        \
        /* the maximum block size in z direction is 64 for all compute capabilities */                              \
        assert(this->_blockDim.z()<=64);                                                                            \
        dim3 blockDim(this->_blockDim.x(), this->_blockDim.y(), this->_blockDim.z());                               \
        kernel::detail::SphericMapper<Zone::dim> mapper;                                                            \
        using namespace PMacc;                                                                                      \
        __cudaKernel(kernel::detail::kernelForeach)(mapper.cudaGridDim(p_zone.size, this->_blockDim), blockDim)      \
                /*   c0_shifted, ..., cN_shifted    */                                                              \
            (mapper, BOOST_PP_ENUM(N, SHIFTED_CURSOR, _), lambda::make_Functor(functor));                           \
    }

/** Foreach algorithm that calls a cuda kernel
 *
 * This is the run-time version of kernel::Foreach where the
 * cuda blockDim is specified in the constructor
 *
 */
struct Foreach
{
    math::Size_t<3> _blockDim;

    /* \param _blockDim size of the cuda blockDim.
     *
     * blockDim has to fit into the computing volume.
     * E.g. (8,8,4) fits into (256, 256, 256)
     *
     */
    Foreach(math::Size_t<3> _blockDim) : _blockDim(_blockDim) {}

    /* operator()(zone, cursor0, cursor1, ..., cursorN-1, functor or lambdaFun)
     *
     * \param zone Accepts currently only a zone::SphericZone object (e.g. containerObj.zone())
     * \param cursorN cursor for the N-th data source (e.g. containerObj.origin())
     * \param functor or lambdaFun either a functor with N arguments or a N-ary lambda function (e.g. _1 = _2)
     *
     * The functor or lambdaFun is called for each cell within the zone.
     * It is called like functor(*cursor0(cellId), ..., *cursorN(cellId))
     *
     */
    BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(FOREACH_KERNEL_MAX_PARAMS), FOREACH_OPERATOR, _)
};

#undef FOREACH_OPERATOR
#undef SHIFT_CURSOR_ZONE
#undef SHIFTED_CURSOR

} // namespace RT
} // namespace kernel
} // namespace algorithm
} // namespace PMacc
