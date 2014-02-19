/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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

#include <iostream>
#include "simulation_defines.hpp"


#include "fields/FieldB.hpp"

#include "fields/LaserPhysics.hpp"

#include "eventSystem/EventSystem.hpp"
#include "dataManagement/DataConnector.hpp"
#include "mappings/kernel/AreaMapping.hpp"
#include "mappings/kernel/ExchangeMapping.hpp"
#include "memory/buffers/GridBuffer.hpp"

#include "fields/FieldManipulator.hpp"

#include "dimensions/SuperCellDescription.hpp"

#include "FieldE.hpp"

#include "MaxwellSolver/Solvers.hpp"
#include "fields/numericalCellTypes/NumericalCellTypes.hpp"

#include "math/vector/compile-time/Vector.hpp"

#include <list>

namespace picongpu
{

using namespace PMacc;

FieldB::FieldB( MappingDesc cellDescription ) :
SimulationFieldHelper<MappingDesc>( cellDescription ),
fieldE( NULL )
{
    /*#####create FieldB###############*/
    fieldB = new GridBuffer<ValueType, simDim > ( cellDescription.getGridLayout( ) );

    /* Calculate the maximum Neighbors we need from MAX(ParticleShape, FieldSolver) */
    typedef typename PMacc::math::CT::max<
        GetMargin<fieldSolver::FieldToParticleInterpolation>::LowerMargin,
        GetMargin<fieldSolver::FieldSolver, FIELD_B>::LowerMargin
        >::type LowerMargin;
    typedef typename PMacc::math::CT::max<
        GetMargin<fieldSolver::FieldToParticleInterpolation>::UpperMargin,
        GetMargin<fieldSolver::FieldSolver, FIELD_B>::UpperMargin
        >::type UpperMargin;

    const DataSpace<simDim> originGuard( LowerMargin( ).vec( ) );
    const DataSpace<simDim> endGuard( UpperMargin( ).vec( ) );

    /*go over all directions*/
    for ( int i = 1; i < numberOfNeighbors[simDim]; ++i )
    {
        DataSpace<simDim> relativMask = Mask::getRelativeDirections<simDim > ( i );
        /* guarding cells depend on direction
         * for negativ direction use originGuard else endGuard (relativ direction ZERO is ignored)
         * * don't switch end and origin because this is a readbuffer and no sendbuffer
         */
        DataSpace<simDim> guardingCells;
        for ( uint32_t d = 0; d < simDim; ++d )
            guardingCells[d] = ( relativMask[d] == -1 ? originGuard[d] : endGuard[d] );
        fieldB->addExchange( GUARD, i, guardingCells, FIELD_B );
    }

}

FieldB::~FieldB( )
{

    delete fieldB;
}

void FieldB::synchronize( )
{
    fieldB->deviceToHost( );
}

void FieldB::syncToDevice( )
{

    fieldB->hostToDevice( );
}

EventTask FieldB::asyncCommunication( EventTask serialEvent )
{

    EventTask eB = fieldB->asyncCommunication( serialEvent );
    return eB;
}

void FieldB::init( FieldE &fieldE, LaserPhysics &laserPhysics )
{

    this->fieldE = &fieldE;
    this->laser = &laserPhysics;

    DataConnector::getInstance( ).registerData( *this, FIELD_B );
}

GridLayout<simDim> FieldB::getGridLayout( )
{

    return cellDescription.getGridLayout( );
}

FieldB::DataBoxType FieldB::getHostDataBox( )
{

    return fieldB->getHostBuffer( ).getDataBox( );
}

FieldB::DataBoxType FieldB::getDeviceDataBox( )
{

    return fieldB->getDeviceBuffer( ).getDataBox( );
}

GridBuffer<FieldB::ValueType, simDim> &FieldB::getGridBuffer( )
{

    return *fieldB;
}

void FieldB::reset( uint32_t )
{
    fieldB->getHostBuffer( ).reset( true );
    fieldB->getDeviceBuffer( ).reset( false );
}

typename FieldB::UnitValueType
FieldB::getUnit( )
{
    return UnitValueType( UNIT_BFIELD, UNIT_BFIELD, UNIT_BFIELD );
}

std::string
FieldB::getName( )
{
    return "FieldB";
}

uint32_t
FieldB::getCommTag( )
{
    return FIELD_B;
}

}
