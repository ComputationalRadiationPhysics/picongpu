/**
 * Copyright 2014 Axel Huebl, Alexander Debus
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

#include "types.h"
#include "simulation_defines.hpp"
#include "basicOperations.hpp"

#include "dimensions/DataSpace.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "mappings/kernel/MappingDescription.hpp"
#include "simulationControl/MovingWindow.hpp"


namespace picongpu
{
namespace cellwiseOperation
{
    using namespace PMacc;

    /** Kernel that calls T_OpFunctor and T_ValFunctor on each cell of a field
     *
     *  Pseudo code: opFunctor( cell, valFunctor( globalCellIdx, currentStep ) );
     *
     * \tparam T_OpFunctor like assign, add, subtract, ...
     * \tparam T_ValFunctor like "f(x,t)", "0.0", "readFromOtherField", ...
     * \tparam FieldBox field type
	 * \tparam halfSimSize specifies the vector to the middle of the global simulation volume
     * \tparam Mapping auto attached argument from __picKernelArea call
     */
    template<
        class T_OpFunctor,
        class T_ValFunctor,
        class FieldBox,
        class Mapping>
    __global__ void
    kernelCellwiseOperation( FieldBox field, T_OpFunctor opFunctor, T_ValFunctor valFunctor, const DataSpace<simDim> totalCellOffset,
        const DataSpace<simDim> halfSimSize, const uint32_t currentStep, Mapping mapper )
    {
        const DataSpace<simDim> block( mapper.getSuperCellIndex( DataSpace<simDim>( blockIdx ) ) );
        const DataSpace<simDim> blockCell = block * MappingDesc::SuperCellSize::toRT();

        const DataSpace<simDim> threadIndex( threadIdx );
		
        opFunctor( field( blockCell + threadIndex ),
                   valFunctor( blockCell + threadIndex + totalCellOffset,
                               currentStep, halfSimSize )
                 );
    }

    /** Call a functor on each cell of a field
     *
     *  \tparam T_Area Where to compute on (CORE, BORDER, GUARD)
     */
    template<uint32_t T_Area>
    class CellwiseOperation
    {
    private:
        typedef MappingDesc::SuperCellSize SuperCellSize;

        MappingDesc cellDescription;

    public:
        CellwiseOperation(MappingDesc cellDescription) : cellDescription(cellDescription)
        {
        }

        /* Functor call to execute the op/valFunctor on a given field
         *
         * \tparam ValFunctor A Value-Producing functor for a given cell
         *                    in time and space
         * \tparam OpFunctor A manipulating functor like PMacc::nvidia::functors::add
         */
        template<class T_Field, class T_OpFunctor, class T_ValFunctor>
        void
        operator()( T_Field field, T_OpFunctor opFunctor, T_ValFunctor valFunctor, uint32_t currentStep, const bool enabled = true ) const
        {
            if( !enabled )
                return;

            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            /** offset due to being the n-th GPU */
            DataSpace<simDim> totalCellOffset(subGrid.getLocalDomain().offset);
            const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter( currentStep );

            /** Assumption: all GPUs have the same number of cells in
             *              y direction for sliding window */
            totalCellOffset.y() += numSlides * subGrid.getLocalDomain().size.y();
            /* the first block will start with less offset if started in the GUARD */
            if( T_Area & GUARD)
                totalCellOffset -= cellDescription.getSuperCellSize() * cellDescription.getGuardingSuperCells();
            /* if we run _only_ in the CORE we have to add the BORDER's offset */
            else if( T_Area == CORE )
                totalCellOffset += cellDescription.getSuperCellSize() * cellDescription.getBorderSuperCells();

			/** Half the size of the global Simulation volume for centering the coordinate origin  */
			const DataSpace<simDim> halfSimSize(subGrid.getGlobalDomain().size / 2); //To DO: In the future, should we also include the globalDomain.offset? For comparison: LaserPhysics.hpp does not do this.			
			
            /* start kernel */
            __picKernelArea((kernelCellwiseOperation<T_OpFunctor>), cellDescription, T_Area)
                    (SuperCellSize::toRT().toDim3())
                    (field->getDeviceDataBox(), opFunctor, valFunctor, totalCellOffset, halfSimSize, currentStep);
        }
    };

} // namespace cellwiseOperation
} // namespace picongpu
