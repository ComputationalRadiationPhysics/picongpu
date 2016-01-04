/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Richard Pausch
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

#include <utility>

#include "cuSTL/cursor/MultiIndexCursor.hpp"
#include "cuSTL/algorithm/cudaBlock/Foreach.hpp"
#include "cuSTL/algorithm/kernel/ForeachBlock.hpp"
#include "cuSTL/container/compile-time/SharedBuffer.hpp"
#include "math/Vector.hpp"
#include "math/VectorOperations.hpp"
#include "particles/access/Cell2Particle.hpp"

#include "PhaseSpace.hpp"

namespace picongpu
{
    using namespace PMacc;

    /** Atomic Add Functor
     *
     * \tparam Type of the values to perform a atomicAdd on
     */
    template<typename Type>
    struct FunctorAtomicAdd
    {
        typedef void result_type;

        DINLINE void
        operator()( Type& dest, const Type src )
        {
            atomicAddWrapper( &dest, src );
        }
    };

    /** Functor called for each particle
     *
     * Every particle in a frame of particles will end up here.
     * We calculate where in space the owning (super) cell lives and
     * add the particle to the shared memory buffer for that phase
     * space snippet the super cell contributes to.
     *
     * \tparam r_dir spatial direction of the phase space (0,1,2) \see AxisDescription
     * \tparam num_pbins number of bins in momentum space \see PhaseSpace.hpp
     * \tparam SuperCellSize how many cells form a super cell \see memory.param
     */
    template<uint32_t r_dir, uint32_t num_pbins, typename SuperCellSize>
    struct FunctorParticle
    {
        typedef void result_type;

        /** Functor implementation
         *
         * \param frame current frame for this block
         * \param particleID id of the particle in the current frame
         * \param curDBufferOriginInBlock section of the phase space, shifted to the start of the block
         * \param el_p coordinate of the momentum \see PhaseSpace::axis_element \see AxisDescription
         * \param axis_p_range range of the momentum coordinate \see PhaseSpace::axis_p_range
         */
        template<typename FramePtr, typename float_PS, typename Pitch >
        DINLINE void
        operator()( FramePtr frame,
                    uint16_t particleID,
                    cursor::CT::BufferCursor<float_PS, Pitch> curDBufferOriginInBlock,
                    const uint32_t el_p,
                    const std::pair<float_X, float_X>& axis_p_range )
        {
            PMACC_AUTO( particle, frame[particleID] );
            /** \todo this can become a functor to be even more flexible */
            const float_X mom_i = particle[momentum_][el_p];

            /* cell id in this block */
            const int linearCellIdx = particle[localCellIdx_];
            const PMacc::math::UInt32<simDim> cellIdx(
                PMacc::math::MapToPos<simDim>()( SuperCellSize(), linearCellIdx ) );

            const uint32_t r_bin    = cellIdx[r_dir];
            const float_X weighting = particle[weighting_];
            const float_X charge    = attribute::getCharge( weighting,particle );
            const float_PS particleChargeDensity =
              precisionCast<float_PS>( charge / CELL_VOLUME );

            const float_X rel_bin = (mom_i - axis_p_range.first)
                                  / (axis_p_range.second - axis_p_range.first);
            int p_bin = int( rel_bin * float_X(num_pbins) );

            /* out-of-range bins back to min/max */
            p_bin >= 0 ? /* do not change p_bin */ : p_bin=0;
            p_bin < num_pbins ? /* do not change p_bin */ : p_bin=num_pbins-1;

            /** \todo take particle shape into account */
            atomicAddWrapper( &(*curDBufferOriginInBlock( p_bin, r_bin )),
                              particleChargeDensity );
        }
    };

    /** Functor to Run For Each SuperCell
     *
     * This functor is called for each super cell, preparing a shared memory
     * buffer with a supercell-local (spatial) snippet of the phase space.
     * Afterwards all blocks reduce their data to a combined gpu-local (spatial)
     * snippet of the phase space in global memory.
     *
     * \tparam Species the particle species to create the phase space for
     * \tparam SuperCellSize how many cells form a super cell \see memory.param
     * \tparam float_PS type for each bin in the phase space
     * \tparam num_pbins number of bins in momentum space \see PhaseSpace.hpp
     * \tparam r_dir spatial direction of the phase space (0,1,2) \see AxisDescription
     */
    template<typename Species, typename SuperCellSize, typename float_PS, uint32_t num_pbins, uint32_t r_dir>
    struct FunctorBlock
    {
        typedef void result_type;

        typedef typename Species::ParticlesBoxType TParticlesBox;

        TParticlesBox particlesBox;
        cursor::BufferCursor<float_PS, 2> curOriginPhaseSpace;
        uint32_t p_element;
        std::pair<float_X, float_X> axis_p_range;

        /** Constructor to transfer params to device
         *
         * \param pb ParticleBox for a species
         * \param cur cursor to start of the local phase space in global memory
         * \param p_dir direction of the 2D phase space in momentum \see AxisDescription
         * \param p_range range of the momentum axis \see PhaseSpace::axis_p_range
         */
        HDINLINE
        FunctorBlock( const TParticlesBox& pb,
                      cursor::BufferCursor<float_PS, 2> cur,
                      const uint32_t p_dir,
                      const std::pair<float_X, float_X>& p_range ) :
        particlesBox(pb), curOriginPhaseSpace(cur), p_element(p_dir),
        axis_p_range(p_range)
        {}

        /** Called for the first cell of each block #-of-cells-in-block times
         *
         * \param indexBlockOffset cell index in global memory, describes where
         *                         the current block starts
         *                         \see cuSTL/algorithm/kernel/ForeachBlock.hpp
         */
        DINLINE void
        operator()( const PMacc::math::Int<simDim>& indexBlockOffset )
        {
            /** \todo write math::Vector constructor that supports dim3 */
            const PMacc::math::Int<simDim> indexGlobal = indexBlockOffset;

            /* create shared mem */
            const int blockCellsInDir = SuperCellSize::template at<r_dir>::type::value;
            typedef typename PMacc::math::CT::Int<num_pbins, blockCellsInDir> dBufferSizeInBlock;
            container::CT::SharedBuffer<float_PS, dBufferSizeInBlock > dBufferInBlock;

            /* init shared mem */
            PMacc::algorithm::cudaBlock::Foreach<SuperCellSize> forEachThreadInBlock;
            {
                using namespace lambda;
                DECLARE_PLACEHOLDERS();
                forEachThreadInBlock( dBufferInBlock.zone(),
                                      dBufferInBlock.origin(),
                                      _1 = float_PS(0.0) );
            }
            __syncthreads();

            FunctorParticle<r_dir, num_pbins, SuperCellSize> functorParticle;
            particleAccess::Cell2Particle<SuperCellSize> forEachParticleInCell;
            forEachParticleInCell( /* mandatory params */
                                   particlesBox, indexGlobal, functorParticle,
                                   /* optional params */
                                   dBufferInBlock.origin(),
                                   p_element,
                                   axis_p_range
                                 );

            __syncthreads();
            /* add to global dBuffer */
            forEachThreadInBlock( /* area to work on */
                                  dBufferInBlock.zone(),
                                  /* data below - cursors will be shifted and
                                   * dereferenced */
                                  curOriginPhaseSpace(0, indexBlockOffset[r_dir]),
                                  dBufferInBlock.origin(),
                                  /* functor */
                                  FunctorAtomicAdd<float_PS>() );
        }
    };

} // namespace picongpu
