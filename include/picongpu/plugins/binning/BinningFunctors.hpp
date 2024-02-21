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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
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

#include <cstdint>

namespace picongpu
{
    namespace plugins::binning
    {
        template<typename T_HistBox, typename T_DepositionFunctor, typename T_AxisTuple, uint32_t N_Axes>
        struct FunctorParticle
        {
            typedef void result_type;

            /** the full histogram data box */
            T_HistBox histBox;
            T_DepositionFunctor quantityFunctor;
            T_AxisTuple axisTuple;
            DomainInfo domainInfo;
            DataSpace<N_Axes> extentsDataspace;

            DINLINE FunctorParticle(
                T_HistBox hBox,
                const T_DepositionFunctor& depositFunc,
                const T_AxisTuple& axes,
                DomainInfo domInfo,
                const DataSpace<N_Axes>& extents)
                : histBox{hBox}
                , quantityFunctor{depositFunc}
                , axisTuple{axes}
                , domainInfo{domInfo}
                , extentsDataspace{extents}
            {
            }

            template<typename T_Worker, typename T_Particle>
            DINLINE void operator()(const T_Worker& worker, const T_Particle& particle)
            {
                using DepositionType = typename T_HistBox::ValueType;

                auto binsDataspace = DataSpace<N_Axes>{};
                bool validIdx = true;
                apply(
                    [&](auto const&... tupleArgs)
                    {
                        uint32_t i = 0;
                        // This assumes n_bins and getBinIdx exist
                        ((binsDataspace[i++] = tupleArgs.getBinIdx(domainInfo, worker, particle, validIdx)), ...);
                    },
                    axisTuple);

                if(validIdx)
                {
                    auto const idxOneD = pmacc::math::linearize(extentsDataspace, binsDataspace);
                    DepositionType depositVal = quantityFunctor(worker, particle);
                    alpaka::atomicAdd(
                        worker.getAcc(),
                        // &(histBox(binsDataspace)),
                        &(histBox(DataSpace<1u>{static_cast<int>(idxOneD)})),
                        depositVal,
                        ::alpaka::hierarchy::Blocks{});
                }
            }
        };

        /**
         * Functor to run on each block on device
         * Allocate memory on device and start iteration over particles
         */
        template<
            typename TParticlesBox,
            typename T_HistBox,
            typename T_DepositionFunctor,
            typename T_AxisTuple,
            uint32_t N_Axes>
        struct FunctorBlock
        {
            using result_type = void;

            TParticlesBox particlesBox;
            T_HistBox binningBox;
            T_DepositionFunctor quantityFunctor;
            T_AxisTuple axisTuple;
            pmacc::DataSpace<SIMDIM> globalOffset;
            pmacc::DataSpace<SIMDIM> localOffset;
            uint32_t currentStep;
            DataSpace<N_Axes> extentsDataspace;

            /** Constructor to transfer params to device
             *
             * @param pb ParticleBox for a species
             */
            HINLINE
            FunctorBlock(
                const TParticlesBox& pBox,
                T_HistBox hBox,
                const T_DepositionFunctor& depositFunc,
                const T_AxisTuple& axes,
                const pmacc::DataSpace<SIMDIM> gOffset,
                const pmacc::DataSpace<SIMDIM> lOffset,
                const uint32_t step,
                const DataSpace<N_Axes>& extents)
                : particlesBox{pBox}
                , binningBox{hBox}
                , quantityFunctor{depositFunc}
                , axisTuple{axes}
                , globalOffset{gOffset}
                , localOffset{lOffset}
                , currentStep{step}
                , extentsDataspace{extents}
            {
            }


            template<typename T_Worker, typename T_Mapping>
            DINLINE void operator()(const T_Worker& worker, T_Mapping const& mapper) const
            {
                const DataSpace<SIMDIM> superCellIdx(mapper.getSuperCellIndex(device::getBlockIdx(worker.getAcc())));

                /**
                 * Init the Domain info, here because of the possibility of a moving window
                 */
                auto domainInfo = DomainInfo{
                    currentStep,
                    globalOffset,
                    localOffset,
                    superCellIdx - mapper.getGuardingSuperCells()};

                FunctorParticle<T_HistBox, T_DepositionFunctor, T_AxisTuple, N_Axes>
                    functorParticle(binningBox, quantityFunctor, axisTuple, domainInfo, extentsDataspace);

                auto forEachParticle
                    = pmacc::particles::algorithm::acc::makeForEach(worker, particlesBox, superCellIdx);

                forEachParticle([&](auto const& lockstepWorker, auto& particle)
                                { functorParticle(lockstepWorker, particle); });
            }
        };
    } // namespace plugins::binning
} // namespace picongpu
