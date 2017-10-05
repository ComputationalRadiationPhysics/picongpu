/* Copyright 2013-2017 Heiko Burau, Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <stdint.h>
#include <pmacc/math/Vector.hpp>
#include <pmacc/math/MapTuple.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <boost/mpl/void.hpp>

namespace particleAccess
{

#define TEMPLATE_ARGS(Z, N, _) typename Arg ## N
#define NORMAL_ARGS(Z, N, _) Arg ## N arg ## N
#define ARGS(Z, N, _) arg ## N

#define CELL2PARTICLE_OPERATOR(Z, N, _) \
template<typename SuperCellSize> \
template<typename T_Acc, typename TParticlesBox, typename CellIndex, typename Functor \
         BOOST_PP_ENUM_TRAILING(N, TEMPLATE_ARGS, _)> \
DINLINE void Cell2Particle<SuperCellSize>::operator() \
(T_Acc const & acc, TParticlesBox pb, const CellIndex& cellIndex, Functor functor \
BOOST_PP_ENUM_TRAILING(N, NORMAL_ARGS, _)) \
{ \
    CellIndex superCellIdx = cellIndex / (CellIndex)SuperCellSize::toRT(); \
    \
    uint16_t linearThreadIdx = threadIdx.z * SuperCellSize::x::value * SuperCellSize::y::value + \
                               threadIdx.y * SuperCellSize::x::value + threadIdx.x; \
    \
    typedef typename TParticlesBox::FramePtr FramePtr; \
    typedef typename TParticlesBox::FrameType Frame; \
    PMACC_SMEM( acc, frame, FramePtr ); \
    PMACC_SMEM( acc, particlesInSuperCell, uint16_t ); \
    \
    if(linearThreadIdx == 0) \
    { \
        frame = pb.getLastFrame(superCellIdx); \
        particlesInSuperCell = pb.getSuperCell(superCellIdx).getSizeLastFrame(); \
    } \
    __syncthreads(); \
    \
    if (!frame.isValid()) return; /* leave kernel if we have no frames*/ \
    \
    while (frame.isValid()) \
    { \
        if (linearThreadIdx < particlesInSuperCell) \
        { \
            functor( \
                acc, \
                frame, linearThreadIdx \
                BOOST_PP_ENUM_TRAILING(N, ARGS, _) \
                ); \
        } \
        __syncthreads(); \
        if (linearThreadIdx == 0) \
        { \
            frame = pb.getPreviousFrame(frame); \
            particlesInSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value; \
        } \
        __syncthreads(); \
    } \
}

BOOST_PP_REPEAT(5, CELL2PARTICLE_OPERATOR, _)

#undef CELL2PARTICLE_OPERATOR
#undef TEMPLATE_ARGS
#undef NORMAL_ARGS
#undef ARGS

}
