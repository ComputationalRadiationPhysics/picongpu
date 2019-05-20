/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch, Benjamin Worpitz, Sergei Bastrakov
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
#include "picongpu/fields/FieldManipulator.hpp"
#include "picongpu/fields/MaxwellSolver/YeePML/Field.hpp"
#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/fields/numericalCellTypes/NumericalCellTypes.hpp"
#include "picongpu/particles/traits/GetInterpolation.hpp"
#include "picongpu/particles/traits/GetMarginPusher.hpp"
#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"

#include <pmacc/dimensions/SuperCellDescription.hpp>
#include <pmacc/eventSystem/EventSystem.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/kernel/ExchangeMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

#include <boost/mpl/accumulate.hpp>

#include <iostream>
#include <list>
#include <memory>


namespace picongpu
{
namespace fields
{
namespace maxwellSolver
{
namespace yeePML
{

    Field::Field( MappingDesc cellDescription ) :
    SimulationFieldHelper< MappingDesc >( cellDescription )
    {
        data.reset(
            new GridBuffer< ValueType, simDim > ( cellDescription.getGridLayout( ) )
        );
    }

    void Field::synchronize( )
    {
        data->deviceToHost( );
    }

    void Field::syncToDevice( )
    {
        data->hostToDevice( );
    }

    EventTask Field::asyncCommunication( EventTask serialEvent )
    {
        EventTask eB = data->asyncCommunication( serialEvent );
        return eB;
    }

    GridLayout< simDim > Field::getGridLayout( )
    {
        return cellDescription.getGridLayout( );
    }

    Field::DataBoxType Field::getHostDataBox( )
    {
        return data->getHostBuffer( ).getDataBox( );
    }

    Field::DataBoxType Field::getDeviceDataBox( )
    {
        return data->getDeviceBuffer( ).getDataBox( );
    }

    GridBuffer< Field::ValueType, simDim > & Field::getGridBuffer( )
    {
        return *data;
    }

    void Field::reset( uint32_t )
    {
        data->getHostBuffer( ).reset( true );
        data->getDeviceBuffer( ).reset( false );
    }

} // namespace yeePML
} // namespace maxwellSolver
} // namespace fields
} // namespace picongpu
