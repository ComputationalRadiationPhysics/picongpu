/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Richard Pausch, Rene Widera
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

#include "picongpu/algorithms/Set.hpp"
#include "picongpu/plugins/PhaseSpace/PhaseSpace.hpp"

#include <pmacc/lockstep.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>

#include <utility>

namespace picongpu
{
    /** Functor called for each particle
     *
     * Every particle in a frame of particles will end up here.
     * We calculate where in space the owning (super) cell lives and
     * add the particle to the shared memory buffer for that phase
     * space snippet the super cell contributes to.
     *
     * @tparam r_dir spatial direction of the phase space (0,1,2) @see AxisDescription
     * @tparam num_pbins number of bins in momentum space @see PhaseSpace.hpp
     */
    template<uint32_t r_dir, uint32_t num_pbins>
    struct FunctorParticle
    {
        typedef void result_type;

        /** Functor implementation
         *
         * @param particle particle to process
         * @param sharedMemHist section of the phase space, shifted to the start of the block
         * @param el_p coordinate of the momentum @see PhaseSpace::axis_element @see AxisDescription
         * @param axis_p_range range of the momentum coordinate @see PhaseSpace::axis_p_range
         */
        template<typename T_Particle, typename T_SharedMemHist, typename T_Worker>
        DINLINE void operator()(
            const T_Worker& worker,
            T_Particle particle,
            T_SharedMemHist sharedMemHist,
            const uint32_t el_p,
            const std::pair<float_X, float_X>& axis_p_range)
        {
            using float_PS = typename T_SharedMemHist::ValueType;
            /** \todo this can become a functor to be even more flexible
             * This requires increasing the y extent of the histogram to take the guard into account
             */
            const float_X mom_i = particle[momentum_][el_p];

            /* cell id in this block */
            const int linearCellIdx = particle[localCellIdx_];
            const pmacc::math::UInt32<simDim> cellIdx(pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx));

            const uint32_t r_bin = cellIdx[r_dir];
            const float_X weighting = particle[weighting_];
            const float_X charge = attribute::getCharge(weighting, particle);
            const float_PS particleChargeDensity = precisionCast<float_PS>(charge / CELL_VOLUME);

            const float_X rel_bin
                = (mom_i / weighting - axis_p_range.first) / (axis_p_range.second - axis_p_range.first);
            int p_bin = int(rel_bin * float_X(num_pbins));

            /* out-of-range bins back to min/max */
            if(p_bin < 0)
                p_bin = 0;
            if(p_bin >= static_cast<int32_t>(num_pbins))
                p_bin = num_pbins - 1;

            /** @todo take particle shape into account */
            cupla::atomicAdd(
                worker.getAcc(),
                &(sharedMemHist(DataSpace<2>(p_bin, r_bin))),
                particleChargeDensity,
                ::alpaka::hierarchy::Threads{});
        }
    };

    /** Functor to Run For Each SuperCell
     *
     * This functor is called for each super cell, preparing a shared memory
     * buffer with a supercell-local (spatial) snippet of the phase space.
     * Afterwards all blocks reduce their data to a combined gpu-local (spatial)
     * snippet of the phase space in global memory.
     *
     * @tparam Species the particle species to create the phase space for
     * @tparam T_filter type of the particle filter
     * @tparam float_PS type for each bin in the phase space
     * @tparam num_pbins number of bins in momentum space @see PhaseSpace.hpp
     * @tparam r_dir spatial direction of the phase space (0,1,2) @see AxisDescription
     */
    template<typename Species, typename float_PS, uint32_t num_pbins, uint32_t r_dir, typename T_Filter>
    struct FunctorBlock
    {
        typedef void result_type;

        typedef typename Species::ParticlesBoxType TParticlesBox;

        TParticlesBox particlesBox;
        pmacc::DataBox<PitchedBox<float_PS, 2>> globalHist;
        uint32_t p_element;
        std::pair<const float_X, float_X> axis_p_range;
        T_Filter particleFilter;

        /** Constructor to transfer params to device
         *
         * @param pb ParticleBox for a species
         * @param parFilter filter functor to select particles
         * @param phaseSpaceHistogram device local part of the histogram
         * @param p_dir direction of the 2D phase space in momentum @see AxisDescription
         * @param p_range range of the momentum axis @see PhaseSpace::axis_p_range
         */
        HDINLINE
        FunctorBlock(
            const TParticlesBox& pb,
            pmacc::DataBox<PitchedBox<float_PS, 2>> phaseSpaceHistogram,
            const uint32_t p_dir,
            const std::pair<float_X, float_X>& p_range,
            const T_Filter& parFilter)
            : particlesBox(pb)
            , globalHist(phaseSpaceHistogram)
            , p_element(p_dir)
            , axis_p_range(p_range)
            , particleFilter(parFilter)
        {
        }

        HDINLINE FunctorBlock(const FunctorBlock&) = default;

        HDINLINE FunctorBlock& operator=(const FunctorBlock&) = default;

        /** Called for the first cell of each block #-of-cells-in-block times
         */
        template<typename T_Worker, typename T_Mapping>
        DINLINE void operator()(const T_Worker& worker, T_Mapping const& mapper) const
        {
            const DataSpace<simDim> superCellIdx(
                mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(worker.getAcc()))));

            /* create shared mem */
            constexpr int blockCellsInDir = SuperCellSize::template at<r_dir>::type::value;
            using SharedMemSize = SuperCellDescription<pmacc::math::CT::Int<num_pbins, blockCellsInDir>>;
            auto sharedMemHist = CachedBox::create<0u, float_PS>(worker, SharedMemSize{});

            Set<float_PS> set(float_PS{0.0});
            auto collectiveOnSharedHistogram = makeThreadCollective<SharedMemSize>();

            /* initialize shared memory with zeros */
            collectiveOnSharedHistogram(worker, set, sharedMemHist);

            worker.sync();

            FunctorParticle<r_dir, num_pbins> functorParticle;

            auto accFilter = particleFilter(worker, superCellIdx - mapper.getGuardingSuperCells());

            auto forEachParticle
                = pmacc::particles::algorithm::acc::makeForEach(worker, particlesBox, DataSpace<simDim>(superCellIdx));

            forEachParticle(
                [&](auto const& lockstepWorker, auto& particle)
                {
                    if(accFilter(lockstepWorker, particle))
                        functorParticle(lockstepWorker, particle, sharedMemHist, p_element, axis_p_range);
                });

            worker.sync();

            // flush shared memory data to global stored histogram
            auto supercellCellOffset = (superCellIdx - mapper.getGuardingSuperCells()) * SuperCellSize::toRT();
            auto const atomicOp = kernel::operation::Atomic<::alpaka::AtomicAdd, ::alpaka::hierarchy::Blocks>{};
            collectiveOnSharedHistogram(
                worker,
                atomicOp,
                globalHist.shift({0, supercellCellOffset[r_dir]}),
                sharedMemHist);
        }
    };

} // namespace picongpu
