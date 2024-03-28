/* Copyright 2014-2023 Axel Huebl, Rene Widera
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/simulation/control/MovingWindow.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>


namespace picongpu
{
    namespace cellwiseOperation
    {
        //! call a functor for each cell
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
             * @tparam T_Worker lockstep worker type
             *
             * @param worker lockstep worker
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
                typename T_Worker>
            DINLINE void operator()(
                T_Worker const& worker,
                T_FieldBox field,
                T_OpFunctor opFunctor,
                T_ValFunctor valFunctor,
                DataSpace<simDim> const totalDomainOffset,
                uint32_t const currentStep,
                T_Mapping mapper) const
            {
                constexpr uint32_t cellsPerSupercell = pmacc::math::CT::volume<SuperCellSize>::type::value;

                DataSpace<simDim> const block(mapper.getSuperCellIndex(worker.blockDomIdxND()));
                DataSpace<simDim> const blockCell = block * SuperCellSize::toRT();
                DataSpace<simDim> const guardCells = mapper.getGuardingSuperCells() * SuperCellSize::toRT();

                lockstep::makeForEach<cellsPerSupercell>(worker)(
                    [&](int32_t const linearIdx)
                    {
                        // cell index within the superCell
                        DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearIdx);

                        opFunctor(
                            worker,
                            field(blockCell + cellIdx),
                            valFunctor(blockCell + cellIdx + totalDomainOffset - guardCells, currentStep));
                    });
            }
        };

        /** Call a functor on each cell of a field
         *
         * @tparam T_Area Where to compute on (CORE, BORDER, GUARD)
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
             * @tparam OpFunctor A manipulating functor like pmacc::math::operation::*
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

                auto const mapper = makeAreaMapper<T_Area>(m_cellDescription);

                PMACC_LOCKSTEP_KERNEL(KernelCellwiseOperation{})
                    .config(mapper.getGridDim(), SuperCellSize{})(
                        field->getDeviceDataBox(),
                        opFunctor,
                        valFunctor,
                        totalDomainOffset,
                        currentStep,
                        mapper);
            }
        };

    } // namespace cellwiseOperation
} // namespace picongpu
