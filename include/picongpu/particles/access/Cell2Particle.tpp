/* Copyright 2013-2021 Heiko Burau, Rene Widera
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
#include <pmacc/mappings/threads/WorkerCfg.hpp>
#include <pmacc/mappings/threads/ForEachIdx.hpp>
#include <pmacc/mappings/threads/IdxConfig.hpp>


namespace picongpu
{
    namespace particleAccess
    {
#define TEMPLATE_ARGS(Z, N, _) typename Arg##N
#define NORMAL_ARGS(Z, N, _) Arg##N arg##N
#define ARGS(Z, N, _) arg##N

#define CELL2PARTICLE_OPERATOR(Z, N, _)                                                                               \
    template<typename SuperCellSize, uint32_t T_numWorkers>                                                           \
    template<                                                                                                         \
        typename T_Acc,                                                                                               \
        typename TParticlesBox,                                                                                       \
        typename CellIndex,                                                                                           \
        typename Functor,                                                                                             \
        typename T_Filter BOOST_PP_ENUM_TRAILING(N, TEMPLATE_ARGS, _)>                                                \
    DINLINE void Cell2Particle<SuperCellSize, T_numWorkers>::operator()(                                              \
        T_Acc const& acc,                                                                                             \
        TParticlesBox pb,                                                                                             \
        const uint32_t workerIdx,                                                                                     \
        const CellIndex& cellIndex,                                                                                   \
        Functor functor,                                                                                              \
        T_Filter filter BOOST_PP_ENUM_TRAILING(N, NORMAL_ARGS, _))                                                    \
    {                                                                                                                 \
        using namespace mappings::threads;                                                                            \
        constexpr uint32_t numWorkers = T_numWorkers;                                                                 \
        constexpr lcellId_t maxParticlesInFrame                                                                       \
            = pmacc::math::CT::volume<typename TParticlesBox::FrameType::SuperCellSize>::type::value;                 \
        CellIndex superCellIdx = cellIndex / (CellIndex) SuperCellSize::toRT();                                       \
                                                                                                                      \
        using FramePtr = typename TParticlesBox::FramePtr;                                                            \
        using Frame = typename TParticlesBox::FrameType;                                                              \
        PMACC_SMEM(acc, frame, FramePtr);                                                                             \
        PMACC_SMEM(acc, particlesInSuperCell, uint16_t);                                                              \
        ForEachIdx<IdxConfig<1, numWorkers>> onlyMaster{workerIdx};                                                   \
                                                                                                                      \
        onlyMaster([&](uint32_t const, uint32_t const) {                                                              \
            frame = pb.getLastFrame(superCellIdx);                                                                    \
            particlesInSuperCell = pb.getSuperCell(superCellIdx).getSizeLastFrame();                                  \
        });                                                                                                           \
        cupla::__syncthreads(acc);                                                                                    \
                                                                                                                      \
        if(!frame.isValid())                                                                                          \
            return; /* leave kernel if we have no frames*/                                                            \
                                                                                                                      \
        auto accFilter                                                                                                \
            = filter(acc, superCellIdx - GuardSize::toRT(), mappings::threads::WorkerCfg<numWorkers>{workerIdx});     \
                                                                                                                      \
        while(frame.isValid())                                                                                        \
        {                                                                                                             \
            using ParticleDomCfg = IdxConfig<maxParticlesInFrame, numWorkers>;                                        \
            ForEachIdx<ParticleDomCfg> forEachParticle(workerIdx);                                                    \
            forEachParticle([&](uint32_t const linearThreadIdx, uint32_t const) {                                     \
                if(linearThreadIdx < particlesInSuperCell)                                                            \
                {                                                                                                     \
                    if(accFilter(acc, frame[linearThreadIdx]))                                                        \
                        functor(acc, frame, linearThreadIdx BOOST_PP_ENUM_TRAILING(N, ARGS, _));                      \
                }                                                                                                     \
            });                                                                                                       \
            cupla::__syncthreads(acc);                                                                                \
            onlyMaster([&](uint32_t const, uint32_t const) {                                                          \
                frame = pb.getPreviousFrame(frame);                                                                   \
                particlesInSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;                           \
            });                                                                                                       \
            cupla::__syncthreads(acc);                                                                                \
        }                                                                                                             \
    }

        BOOST_PP_REPEAT(5, CELL2PARTICLE_OPERATOR, _)

#undef CELL2PARTICLE_OPERATOR
#undef TEMPLATE_ARGS
#undef NORMAL_ARGS
#undef ARGS

    } // namespace particleAccess
} // namespace picongpu
