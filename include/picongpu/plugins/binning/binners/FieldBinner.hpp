/* Copyright 2023-2025 Tapish Narwal
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

// required becuase the definition of Binner is conditionally included
#if (ENABLE_OPENPMD == 1)

#    include "picongpu/plugins/binning/BinningFunctors.hpp"
#    include "picongpu/plugins/binning/binners/Binner.hpp"
#    include "picongpu/plugins/binning/utility.hpp"

#    include <pmacc/math/operation/traits.hpp>
#    include <pmacc/memory/tuple/STLTuple.hpp>
#    include <pmacc/meta/errorHandlerPolicies/ReturnType.hpp>
#    include <pmacc/mpi/MPIReduce.hpp>
#    include <pmacc/mpi/reduceMethods/Reduce.hpp>

#    include <cstdint>

namespace picongpu
{
    namespace plugins::binning
    {
        template<typename TBinningData>
        class FieldBinner : public Binner<TBinningData>
        {
        public:
            FieldBinner(TBinningData const& bd, MappingDesc* cellDesc) : Binner<TBinningData>(bd, cellDesc)
            {
            }

        private:
            using ReductionOp = typename Binner<TBinningData>::ReductionOp;

            void doBinning(uint32_t currentStep) override
            {
                // Call fill fields function
                this->binningData.hostHook();

                // Do binning for each cell. Writes to histBuffer

                // @todo do field filtering?

                auto binningBox = this->histBuffer->getDeviceBuffer().getDataBox();

                auto cellDesc = *this->cellDescription;
                auto const mapper = makeAreaMapper<pmacc::type::CORE + pmacc::type::BORDER>(cellDesc);

                auto const globalOffset = Environment<simDim>::get().SubGrid().getGlobalDomain().offset;
                auto const localOffset = Environment<simDim>::get().SubGrid().getLocalDomain().offset;
                auto& mv = MovingWindow::getInstance();
                auto const windowOffset = mv.getMovingWindowOriginPositionCells(currentStep);
                auto domainInfo = DomainInfo<BinningType::Field>(
                    currentStep,
                    globalOffset,
                    localOffset,
                    mapper.getGuardingSuperCells(),
                    windowOffset);

                auto const axisKernels = pmacc::memory::tuple::tupleMap(
                    this->binningData.axisTuple,
                    [&](auto const& axis) -> decltype(auto) { return axis.getAxisKernel(); });

                auto const functorBlock
                    = FieldBinningKernel<pmacc::math::operation::traits::AlpakaAtomicOp_t<ReductionOp>>{};

                auto const userFunctorData = std::apply(
                    [&](auto&&... fields)
                    {
                        return std::apply(
                            [&](auto&&... extras)
                            {
                                return pmacc::memory::tuple::make_tuple(
                                    transformFieldInfo(std::forward<decltype(fields)>(fields))...,
                                    std::forward<decltype(extras)>(extras)...);
                            },
                            this->binningData.extraData);
                    },
                    this->binningData.fieldsTuple);

                PMACC_LOCKSTEP_KERNEL(functorBlock)
                    .config(mapper.getGridDim(), SuperCellSize{})(
                        binningBox,
                        domainInfo,
                        axisKernels,
                        userFunctorData,
                        this->binningData.depositionData.functor,
                        this->binningData.axisExtentsND,
                        mapper);
            }
        };

    } // namespace plugins::binning
} // namespace picongpu

#endif
