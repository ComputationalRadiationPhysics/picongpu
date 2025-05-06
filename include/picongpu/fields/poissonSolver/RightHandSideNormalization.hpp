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

#include <pmacc/device/Reduce.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/ExchangeMapping.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
#include <pmacc/memory/boxes/DataBoxUnaryTransform.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>

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
        mpi::MPIReduce mpiReduce;
        std::unique_ptr<pmacc::device::Reduce> localReduce;

        RightHandSideNormalization() : localReduce{std::make_unique<pmacc::device::Reduce>(1024)}
        {
            mpiReduce.participate(true);
        }

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

        auto calcNorm(FieldTmp& fieldRho)
        {
            /*define stacked DataBox's for reduce algorithm*/
            using TransformedBox = DataBoxUnaryTransform<typename FieldTmp::DataBoxType, squareComponentWise>;
            using Box64bit = DataBoxUnaryTransform<TransformedBox, cast64Bit>;
            using D1Box = DataBoxDim1Access<Box64bit>;

            /* reduce field E*/
            DataSpace<simDim> fieldSize = fieldRho.getGridLayout().sizeWithoutGuardND();
            DataSpace<simDim> fieldGuard = fieldRho.getGridLayout().guardSizeND();

            TransformedBox fieldTransform(fieldRho.getDeviceDataBox().shift(fieldGuard));
            Box64bit field64bit(fieldTransform);
            D1Box d1Access(field64bit, fieldSize);

            float_64 fieldRhoNormSquaredLocal
                = (*localReduce)(pmacc::math::operation::Add(), d1Access, fieldSize.productOfComponents()).x();

            // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
            eventSystem::getTransactionEvent().waitForFinished();
            float_64 fieldRhoNormSquaredGlobal;
            mpiReduce(
                pmacc::math::operation::Add(),
                &fieldRhoNormSquaredGlobal,
                &fieldRhoNormSquaredLocal,
                1,
                mpi::reduceMethods::AllReduce());

            return math::sqrt(fieldRhoNormSquaredGlobal);
        }

    private:
        template<typename T_Type>
        struct cast64Bit
        {
            using result = typename TypeCast<float_64, T_Type>::result;

            HDINLINE result operator()(T_Type const& value) const
            {
                return precisionCast<float_64>(value);
            }
        };

        template<typename T_Type>
        struct squareComponentWise
        {
            using result = T_Type;

            HDINLINE result operator()(T_Type const& value) const
            {
                return value * value;
            }
        };
    };
} // namespace picongpu::fields::poissonSolver
