/* Copyright 2025-2026 Tapish Narwal, Luca Pennati, Rene Widera, Edgar Marquardt
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
    struct BoundaryFunction
    {
        // gives the potential at the boundary, given the total cell coordinate for Dirichlet BCs and the outward
        // normal derivative for Neumann BCs change this function to implement different boundary conditions. for
        // Dirichlet BCs use units of potential, for Neumann BCs use units of potential per length. maybe make this a
        // param file parameter in the future
        HDINLINE auto operator()(
            math::Vector<double, 3u> const& totalCellCoordinate,
            DataSpace<3u> const& outwardNormal) const
        {
            return 0.0;
        }

        HDINLINE auto operator()(
            math::Vector<double, 2u> const& totalCellCoordinate,
            DataSpace<2u> const& outwardNormal) const
        {
            return 0.0;
        }
    };

    struct ApplyDirichletBCsFromFunctionKernel
    {
        DINLINE auto operator()(
            auto const& worker,
            auto fieldVBox,
            auto const boundaryFunction,
            DataSpace<simDim> cellOffsetToTotalOrigin,
            float_64 const& normalizationFactor,
            auto const mapper) const -> void
        {
            auto const relExchangeDir = Mask::getRelativeDirections<simDim>(mapper.getExchangeType());
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

                    fieldVBox(superCellIdx * SuperCellSize::toRT() + cellIdx)
                        = normalizationFactor * boundaryFunction(totalDistance, relExchangeDir);
                });
        }
    };

    struct ApplyDirichletBCsHomogeneousKernel
    {
        DINLINE auto operator()(
            auto const& worker,
            auto fieldVBox,
            auto const boundaryFunction,
            DataSpace<simDim> cellOffsetToTotalOrigin,
            float_64 const& normalizationFactor,
            auto const mapper) const -> void
        {
            // including guards
            DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));

            constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

            auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);

            forEachCellInSupercell(
                [&](int32_t const linearCellIdx)
                {
                    /* cell index within the superCell */
                    DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);

                    fieldVBox(superCellIdx * SuperCellSize::toRT() + cellIdx) = 0.0;
                });
        }
    };

    struct ApplyNeumannBCsFromFunctionKernel
    {
        DINLINE auto operator()(
            auto const& worker,
            auto fieldVBox,
            auto const boundaryFunction,
            DataSpace<simDim> cellOffsetToTotalOrigin,
            float_64 const& normalizationFactor,
            auto const mapper) const -> void
        {
            auto const relExchangeDir = Mask::getRelativeDirections<simDim>(mapper.getExchangeType());
            // including guards
            DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));

            DataSpace<simDim> const guardCells = mapper.getGuardingSuperCells() * SuperCellSize::toRT();

            // no guards included
            DataSpace<simDim> const localCells = mapper.getGridSuperCells() * SuperCellSize::toRT() - 2 * guardCells;
            DataSpace<simDim> const superCellCellOffset = superCellIdx * SuperCellSize::toRT();

            constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

            auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);

            forEachCellInSupercell(
                [&](int32_t const linearCellIdx)
                {
                    /* cell index within the superCell */
                    DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);
                    // without guards
                    DataSpace<simDim> const totalCellIdx
                        = cellOffsetToTotalOrigin + superCellCellOffset - guardCells + cellIdx;

                    auto totalDistance = precisionCast<float_64>(totalCellIdx)
                                         * precisionCast<float_64>(sim.pic.getCellSize().shrink<simDim>());

                    DataSpace<simDim> const guardCellIdx = superCellCellOffset + cellIdx;
                    DataSpace<simDim> sourceCellIdx = guardCellIdx;

                    uint32_t dir = simDim;
                    for(uint32_t d = 0; d < simDim; ++d)
                    {
                        if(relExchangeDir[d] < 0)
                        {
                            sourceCellIdx[d] = guardCells[d];
                            dir = d;
                        }
                        else if(relExchangeDir[d] > 0)
                        {
                            sourceCellIdx[d] = guardCells[d] + localCells[d] - 1;
                            dir = d;
                        }
                    }

                    if(dir < simDim)
                    {
                        fieldVBox(guardCellIdx) = fieldVBox(sourceCellIdx)
                                                  + normalizationFactor
                                                        * boundaryFunction(totalDistance, relExchangeDir)
                                                        * sim.pic.getCellSize().shrink<simDim>()[dir];
                    }
                    else
                    {
                        fieldVBox(guardCellIdx) = fieldVBox(guardCellIdx);
                    }
                });
        }
    };

    struct ApplyNeumannBCsHomogeneousKernel
    {
        DINLINE auto operator()(
            auto const& worker,
            auto fieldVBox,
            auto const boundaryFunction,
            DataSpace<simDim> cellOffsetToTotalOrigin,
            float_64 const& normalizationFactor,
            auto const mapper) const -> void
        {
            auto const relExchangeDir = Mask::getRelativeDirections<simDim>(mapper.getExchangeType());
            // including guards
            DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));

            DataSpace<simDim> const guardCells = mapper.getGuardingSuperCells() * SuperCellSize::toRT();

            // no guards included
            DataSpace<simDim> const localCells = mapper.getGridSuperCells() * SuperCellSize::toRT() - 2 * guardCells;
            DataSpace<simDim> const superCellCellOffset = superCellIdx * SuperCellSize::toRT();

            constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

            auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);

            forEachCellInSupercell(
                [&](int32_t const linearCellIdx)
                {
                    /* cell index within the superCell */
                    DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);


                    DataSpace<simDim> const guardCellIdx = superCellCellOffset + cellIdx;
                    DataSpace<simDim> sourceCellIdx = guardCellIdx;

                    for(uint32_t d = 0; d < simDim; ++d)
                    {
                        if(relExchangeDir[d] < 0)
                        {
                            sourceCellIdx[d] = guardCells[d];
                        }
                        else if(relExchangeDir[d] > 0)
                        {
                            sourceCellIdx[d] = guardCells[d] + localCells[d] - 1;
                        }
                    }
                    fieldVBox(guardCellIdx) = fieldVBox(sourceCellIdx);
                });
        }
    };

    template<typename T_Kernel, typename T_Buffer>
    void applyBoundaryConditions(
        T_Buffer& buffer,
        MappingDesc const& cellDescription,
        float_64 const& normalizationFactor = 1.0)
    {
        SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
        auto const globalDomain = subGrid.getGlobalDomain();
        auto const localDomain = subGrid.getLocalDomain();
        auto const cellOffsetToTotalOrigin = globalDomain.offset + localDomain.offset;

        for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
        {
            /* only call for planes: left, right, top, bottom, back, front */
            if(FRONT % i == 0 && !Environment<simDim>::get().GridController().getCommunicationMask().isSet(i))
            {
                ExchangeMapping<GUARD, MappingDesc> mapper(cellDescription, i);

                PMACC_LOCKSTEP_KERNEL(T_Kernel{})
                    .config(mapper.getGridDim(), SuperCellSize{})(
                        buffer.getDeviceBuffer().getDataBox(),
                        BoundaryFunction{},
                        cellOffsetToTotalOrigin,
                        normalizationFactor,
                        mapper);
            }
        }
    }

    struct BoundaryConditionsDirichlet
    {
        // Solver work buffers use homogeneous BCs; FieldV uses BoundaryFunction below.
        template<typename T_Buffer>
        void operator()(T_Buffer& buffer, MappingDesc const& cellDescription) const
        {
            applyBoundaryConditions<ApplyDirichletBCsHomogeneousKernel>(buffer, cellDescription);
        }

        void operator()(FieldV& fieldV, MappingDesc const& cellDescription, float_64 const& normalizationFactor = 1.0)
            const
        {
            applyBoundaryConditions<ApplyDirichletBCsFromFunctionKernel>(
                *fieldV.fieldVBuffer,
                cellDescription,
                normalizationFactor);
        }
    };

    struct BoundaryConditionsNeumann
    {
        // Solver work buffers use homogeneous BCs; FieldV uses BoundaryFunction below.
        template<typename T_Buffer>
        void operator()(T_Buffer& buffer, MappingDesc const& cellDescription) const
        {
            applyBoundaryConditions<ApplyNeumannBCsHomogeneousKernel>(buffer, cellDescription);
        }

        void operator()(FieldV& fieldV, MappingDesc const& cellDescription, float_64 const& normalizationFactor = 1.0)
            const
        {
            applyBoundaryConditions<ApplyNeumannBCsFromFunctionKernel>(
                *fieldV.fieldVBuffer,
                cellDescription,
                normalizationFactor);
        }
    };

    // Use Neumann boundary conditions for the Poisson solver by default.
    // Change this to BoundaryConditionsDirichlet to use Dirichlet boundary conditions instead.
    // maybe make this a param file parameter in the future
    using BoundaryConditions = BoundaryConditionsNeumann;
} // namespace picongpu::fields::poissonSolver
