/* Copyright 2025 Tapish Narwal, Luca Pennati, Rene Widera
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

#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldTmpOperations.hpp"
#include "picongpu/fields/poissonSolver/FieldV.hpp"

#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/ExchangeMapping.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>

namespace picongpu::fields::poissonSolver
{
    struct RightHandSideNormalizationKernel
    {
        DINLINE auto operator()(auto const& worker, auto const fieldVBox, auto fieldRhoBox, auto const mapper) const
            -> void
        {
            DataSpace<simDim> const relExchangeDir = Mask::getRelativeDirections<simDim>(mapper.getExchangeType());
            DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));
            DataSpace<simDim> superCellCellOffset = superCellIdx * SuperCellSize::toRT();
            DataSpace<simDim> cellOffsetInDomain
                = superCellCellOffset - mapper.getGuardingSuperCells() * SuperCellSize::toRT();

            DataSpace<simDim> numGuardCells = mapper.getGuardingSuperCells() * SuperCellSize::toRT();
            DataSpace<simDim> adjustedSuperCellCellOffset = superCellCellOffset - (relExchangeDir * numGuardCells);

            DataSpace<simDim> numCellsLocalDomain = mapper.getGuardingSuperCells() * SuperCellSize::toRT();

            constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

            auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);

            forEachCellInSupercell(
                [&](int32_t const linearCellIdx)
                {
                    /* cell index within the superCell */
                    DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);
                    DataSpace<simDim> const localCellIdx = adjustedSuperCellCellOffset + cellIdx;

                    for(uint32_t d = 0; d < simDim; ++d)
                    {
                        if(relExchangeDir[d] != 0
                           && (localCellIdx[d] == 0u || localCellIdx[d] == numCellsLocalDomain[d] - 1))
                        {
                            fieldRhoBox(localCellIdx + numGuardCells)
                                += fieldVBox(localCellIdx + numGuardCells + relExchangeDir)
                                   / (sim.pic.getCellSize()[d] * sim.pic.getCellSize()[d]);
                        }
                    }
                });
        }
    };

    struct RightHandSideNormalization
    {
        // return residual
        // return number of iterations
        void operator()(FieldV& fieldV, FieldTmp& fieldRho, MappingDesc cellDescription)
        {
            /* only call for planes: left right top bottom back front*/
            for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
            {
                if(FRONT % i == 0 && !(Environment<simDim>::get().GridController().getCommunicationMask().isSet(i)))
                {
                    ExchangeMapping<GUARD, MappingDesc> mapper(cellDescription, i);

                    PMACC_LOCKSTEP_KERNEL(RightHandSideNormalizationKernel{})
                        .config(mapper.getGridDim(), SuperCellSize{})(
                            fieldV.fieldVBuffer->getDeviceBuffer().getDataBox(),
                            fieldRho.getDeviceDataBox(),
                            mapper);
                }
            }
        }
    };
} // namespace picongpu::fields::poissonSolver
