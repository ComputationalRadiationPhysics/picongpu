/* Copyright 2014-2021 Axel Huebl, Rene Widera
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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/simulation/control/MovingWindow.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>
#include <pmacc/mappings/threads/ForEachIdx.hpp>
#include <pmacc/mappings/threads/IdxConfig.hpp>


namespace picongpu
{
    namespace cellwiseOperation
    {
        /** call a functor for each cell
         *
         * @tparam T_numWorkers number of workers
         */
        template<uint32_t T_numWorkers>
        struct KernelCellwiseOperation
        {
            /** Kernel that calls T_OpFunctor and T_ValFunctor on each cell of a field
             *
             * performed code for each cell:
             * @code{.cpp}
             * opFunctor( acc, field, valFunctor( totalCellIdx, currentStep ) );
             * @endcode
             *
             * @tparam T_OpFunctor like assign, add, subtract, ...
             * @tparam T_ValFunctor like "f(x,t)", "0.0", "readFromOtherField", ...
             * @tparam T_FieldBox field type
             * @tparam T_Mapping mapper which defines the working region
             * @tparam T_Acc alpaka accelerator type
             *
             * @param acc alpaka accelerator
             * @param[in,out] field field to manipulate
             * @param opFunctor binary operator used with the old and functor value
             *                  (collective functors are not supported)
             * @param valFunctor functor to execute (collective functors are not supported)
             * @param totalDomainOffset offset to the local domain relative to the origin of the global domain
             * @param currentStep simulation time step
             * @param mapper functor to map a block to a supercell
             */
            template<
                typename T_OpFunctor,
                typename T_ValFunctor,
                typename T_FieldBox,
                typename T_Mapping,
                typename T_Acc>
            DINLINE void operator()(
                T_Acc const& acc,
                T_FieldBox field,
                T_OpFunctor opFunctor,
                T_ValFunctor valFunctor,
                DataSpace<simDim> const totalDomainOffset,
                uint32_t const currentStep,
                T_Mapping mapper) const
            {
                using namespace mappings::threads;
                constexpr uint32_t cellsPerSupercell = pmacc::math::CT::volume<SuperCellSize>::type::value;
                constexpr uint32_t numWorker = T_numWorkers;

                uint32_t const workerIdx = cupla::threadIdx(acc).x;

                DataSpace<simDim> const block(mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(acc))));
                DataSpace<simDim> const blockCell = block * SuperCellSize::toRT();
                DataSpace<simDim> const guardCells = mapper.getGuardingSuperCells() * SuperCellSize::toRT();

                ForEachIdx<IdxConfig<cellsPerSupercell, numWorker>>{workerIdx}(
                    [&](uint32_t const linearIdx, uint32_t const) {
                        // cell index within the superCell
                        DataSpace<simDim> const cellIdx
                            = DataSpaceOperations<simDim>::template map<SuperCellSize>(linearIdx);

                        opFunctor(
                            acc,
                            field(blockCell + cellIdx),
                            valFunctor(blockCell + cellIdx + totalDomainOffset - guardCells, currentStep));
                    });
            }
        };

        /** Call a functor on each cell of a field
         *
         * \tparam T_Area Where to compute on (CORE, BORDER, GUARD)
         */
        template<uint32_t T_Area>
        class CellwiseOperation
        {
        private:
            MappingDesc m_cellDescription;

        public:
            CellwiseOperation(MappingDesc const cellDescription) : m_cellDescription(cellDescription)
            {
            }

            /** Functor call to execute the op/valFunctor on a given field
             *
             * @tparam ValFunctor A Value-Producing functor for a given cell
             *                    in time and space
             * @tparam OpFunctor A manipulating functor like pmacc::nvidia::functors::add
             */
            template<typename T_Field, typename T_OpFunctor, typename T_ValFunctor>
            void operator()(T_Field field, T_OpFunctor opFunctor, T_ValFunctor valFunctor, uint32_t const currentStep)
                const
            {
                SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                // offset to the local domain relative to the origin of the global domain
                DataSpace<simDim> totalDomainOffset(subGrid.getLocalDomain().offset);
                uint32_t const numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);

                /** Assumption: all GPUs have the same number of cells in
                 *              y direction for sliding window
                 */
                totalDomainOffset.y() += numSlides * subGrid.getLocalDomain().size.y();

                constexpr uint32_t numWorkers
                    = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

                AreaMapping<T_Area, MappingDesc> mapper(m_cellDescription);

                PMACC_KERNEL(KernelCellwiseOperation<numWorkers>{})
                (mapper.getGridDim(),
                 numWorkers)(field->getDeviceDataBox(), opFunctor, valFunctor, totalDomainOffset, currentStep, mapper);
            }
        };

    } // namespace cellwiseOperation
} // namespace picongpu
