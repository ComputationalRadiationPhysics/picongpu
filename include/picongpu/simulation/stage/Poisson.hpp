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
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/FieldJ.kernel"
#include "picongpu/fields/FieldTmpOperations.hpp"
#include "picongpu/fields/currentDeposition/Deposit.hpp"
#include "picongpu/fields/poissonSolver/FieldV.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/device/Reduce.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

namespace picongpu::simulation::stage
{
    //! Functor for the stage of the PIC loop performing charge deposition
    struct Poisson
    {
        std::unique_ptr<GridBuffer<float_64, simDim>> pkBuffer;
        std::unique_ptr<GridBuffer<float_64, simDim>> rkBuffer;
        std::unique_ptr<GridBuffer<float_64, simDim>> r0Buffer;
        std::unique_ptr<GridBuffer<float_64, simDim>> mpkBuffer;
        std::unique_ptr<GridBuffer<float_64, simDim>> ampkBuffer;
        std::unique_ptr<GridBuffer<float_64, simDim>> zkBuffer;
        std::unique_ptr<GridBuffer<float_64, simDim>> azkBuffer;

        std::unique_ptr<GridBuffer<float_64, simDim>> yBuffer;
        std::unique_ptr<GridBuffer<float_64, simDim>> wBuffer;
        std::unique_ptr<GridBuffer<float_64, simDim>> zBuffer;

        std::shared_ptr<fields::poissonSolver::FieldV> fieldV;

        picongpu::MappingDesc const m_mappingDesc;

        mpi::MPIReduce mpiReduce;
        std::unique_ptr<pmacc::device::Reduce> localReduce;


        Poisson(picongpu::MappingDesc const mappingDesc);
        /** Compute the current created by particles and add it to the current
         *  density
         *
         * @param currentStep index of time iteration
         */
        void operator()(uint32_t const currentStep);

        void participate(bool status)
        {
            mpiReduce.participate(status);
        }

        void preconditioner(
            std::unique_ptr<GridBuffer<float_64, simDim>>& xBuffer,
            std::unique_ptr<GridBuffer<float_64, simDim>>& bBuffer);

    private:
        auto reduceGlobal(DataSpace<simDim> fieldSize, auto dataBoxIn);
    };
} // namespace picongpu::simulation::stage
