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
#include <pmacc/memory/dataTypes/Mask.hpp>

namespace picongpu::fields::poissonSolver
{
    struct SolutionFunction
    {
        HDINLINE auto operator()(math::Vector<double, simDim> const& totalCellCoordinate) const
        {
            if constexpr(simDim == 3u)
            {
                return math::sin(totalCellCoordinate.x()) + math::cos(totalCellCoordinate.y())
                       + 3.0 * math::sin(totalCellCoordinate.z())
                       + totalCellCoordinate.x() * totalCellCoordinate.productOfComponents() + 10.0;
            }
            else if constexpr(simDim == 2u)
            {
                return math::sin(totalCellCoordinate.x()) + math::cos(totalCellCoordinate.y())
                       + totalCellCoordinate.x() * totalCellCoordinate.productOfComponents() + 10.0;
            }
        }
    };

    struct ApplyDirichletBCsFromFunctionKernel
    {
        DINLINE auto operator()(
            auto const& worker,
            auto fieldVBox,
            auto const boundaryFunction,
            DataSpace<simDim> cellOffsetToTotalOrigin,
            auto const mapper) const -> void
        {
            // including guards
            DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));

            DataSpace<simDim> numGuardCells = mapper.getGuardingSuperCells() * SuperCellSize::toRT();

            // no guards included
            DataSpace<simDim> superCellTotalCellOffset
                = cellOffsetToTotalOrigin + superCellIdx * SuperCellSize::toRT() - numGuardCells;

            constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

            auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);

            forEachCellInSupercell(
                [&](int32_t const linearCellIdx)
                {
                    /* cell index within the superCell */
                    DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);
                    // without guards
                    DataSpace<simDim> const totalCellIdx = superCellTotalCellOffset + cellIdx;

                    auto totalDistance = precisionCast<float_64>(totalCellIdx)
                                         * precisionCast<float_64>(sim.pic.getCellSize().shrink<simDim>());

                    fieldVBox(superCellIdx * SuperCellSize::toRT() + cellIdx) = boundaryFunction(totalDistance);
                });
        }
    };

    struct BoundaryConditionsDirichlet
    {
        // return residual
        // return number of iterations
        void operator()(FieldV& fieldV, MappingDesc cellDescription) const
        {
            SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
            auto globalDomain = subGrid.getGlobalDomain();
            auto localDomain = subGrid.getLocalDomain();

            auto cellOffsetToTotalOrigin = globalDomain.offset + localDomain.offset;


            for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
            {
                /* only call for planes: left right top bottom back front*/
                if(FRONT % i == 0 && !(Environment<simDim>::get().GridController().getCommunicationMask().isSet(i)))
                {
                    ExchangeMapping<GUARD, MappingDesc> mapper(cellDescription, i);

                    PMACC_LOCKSTEP_KERNEL(ApplyDirichletBCsFromFunctionKernel{})
                        .config(mapper.getGridDim(), SuperCellSize{})(
                            fieldV.fieldVBuffer->getDeviceBuffer().getDataBox(),
                            SolutionFunction{},
                            cellOffsetToTotalOrigin,
                            mapper);
                }
            }
        }
    };
} // namespace picongpu::fields::poissonSolver
