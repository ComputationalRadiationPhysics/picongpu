/* Copyright 2023 Tapish Narwal
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/plugins/binning/Axis.hpp"
#include "picongpu/plugins/binning/BinningData.hpp"
#include "picongpu/plugins/binning/DomainInfo.hpp"
#include "picongpu/plugins/binning/utility.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>

#include <cstdint>

namespace picongpu
{
    namespace plugins::binning
    {
        struct FunctorParticle
        {
            using result_type = void;

            DINLINE FunctorParticle() = default;

            template<
                typename T_Worker,
                typename T_HistBox,
                typename T_DepositionFunctor,
                typename T_AxisTuple,
                typename T_Particle,
                uint32_t T_nAxes>
            DINLINE void operator()(
                T_Worker const& worker,
                T_HistBox histBox,
                T_DepositionFunctor const& quantityFunctor,
                T_AxisTuple const& axes,
                DomainInfo const& domainInfo,
                DataSpace<T_nAxes> const& extents,
                const T_Particle& particle) const
            {
                using DepositionType = typename T_HistBox::ValueType;

                auto binsDataspace = DataSpace<T_nAxes>{};
                bool validIdx = true;
                apply(
                    [&](auto const&... tupleArgs)
                    {
                        uint32_t i = 0;
                        // This assumes n_bins and getBinIdx exist
                        ((binsDataspace[i++] = tupleArgs.getBinIdx(domainInfo, worker, particle, validIdx)), ...);
                    },
                    axes);

                if(validIdx)
                {
                    auto const idxOneD = pmacc::math::linearize(extents, binsDataspace);
                    DepositionType depositVal = quantityFunctor(worker, particle);
                    alpaka::atomicAdd(worker.getAcc(), &(histBox[idxOneD]), depositVal, ::alpaka::hierarchy::Blocks{});
                }
            }
        };

        /** Creates a histogram based on axis and quantity description
         */
        struct BinningFunctor
        {
            using result_type = void;

            HINLINE BinningFunctor() = default;

            template<
                typename T_Worker,
                typename TParticlesBox,
                typename T_HistBox,
                typename T_DepositionFunctor,
                typename T_AxisTuple,
                typename T_Mapping,
                uint32_t T_nAxes>
            DINLINE void operator()(
                T_Worker const& worker,
                T_HistBox binningBox,
                TParticlesBox particlesBox,
                pmacc::DataSpace<simDim> const& localOffset,
                pmacc::DataSpace<simDim> const& globalOffset,
                T_AxisTuple const& axisTuple,
                T_DepositionFunctor const& quantityFunctor,
                DataSpace<T_nAxes> const& extents,
                uint32_t const currentStep,
                T_Mapping const& mapper) const
            {
                DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));

                /**
                 * Init the Domain info, here because of the possibility of a moving window
                 */
                auto const domainInfo = DomainInfo{
                    currentStep,
                    globalOffset,
                    localOffset,
                    superCellIdx - mapper.getGuardingSuperCells()};

                auto const functorParticle = FunctorParticle{};

                auto forEachParticle
                    = pmacc::particles::algorithm::acc::makeForEach(worker, particlesBox, superCellIdx);

                forEachParticle(
                    [&](auto const& lockstepWorker, auto& particle) {
                        functorParticle(
                            lockstepWorker,
                            binningBox,
                            quantityFunctor,
                            axisTuple,
                            domainInfo,
                            extents,
                            particle);
                    });
            }
        };
    } // namespace plugins::binning
} // namespace picongpu
