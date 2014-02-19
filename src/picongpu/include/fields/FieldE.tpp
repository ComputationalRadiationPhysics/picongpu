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
 


#include "memory/buffers/GridBuffer.hpp"
#include "mappings/simulation/GridController.hpp"

#include "dataManagement/DataConnector.hpp"

#include "mappings/kernel/AreaMapping.hpp"
#include "eventSystem/EventSystem.hpp"
#include "mappings/kernel/ExchangeMapping.hpp"

#include "fields/FieldManipulator.hpp"
#include "dimensions/SuperCellDescription.hpp"

#include "FieldB.hpp"

#include "fields/FieldE.kernel"

#include "MaxwellSolver/Solvers.hpp"
#include "fields/numericalCellTypes/NumericalCellTypes.hpp"

#include "math/vector/compile-time/Vector.hpp"

#include <list>

namespace picongpu
{
using namespace PMacc;

FieldE::FieldE( MappingDesc cellDescription ) :
SimulationFieldHelper<MappingDesc>( cellDescription ),
fieldB( NULL )
{
    fieldE = new GridBuffer<ValueType, simDim > ( cellDescription.getGridLayout( ) );

    /* Calculate the maximum Neighbors we need from MAX(ParticleShape, FieldSolver) */
    typedef typename PMacc::math::CT::max<
        GetMargin<fieldSolver::FieldToParticleInterpolation>::LowerMargin,
        GetMargin<fieldSolver::FieldSolver, FIELD_E>::LowerMargin
        >::type LowerMargin;
    typedef typename PMacc::math::CT::max<
        GetMargin<fieldSolver::FieldToParticleInterpolation>::UpperMargin,
        GetMargin<fieldSolver::FieldSolver, FIELD_E>::UpperMargin
        >::type UpperMargin;

    const DataSpace<simDim> originGuard( LowerMargin( ).vec( ) );
    const DataSpace<simDim> endGuard( UpperMargin( ).vec( ) );

    /*receive from all directions*/
    for ( uint32_t i = 1; i < numberOfNeighbors[simDim]; ++i )
    {
        DataSpace<simDim> relativMask = Mask::getRelativeDirections<simDim > ( i );
        /*guarding cells depend on direction
         * for negativ direction use originGuard else endGuard (relativ direction ZERO is ignored)
         * don't switch end and origin because this is a readbuffer and no sendbuffer 
         */
        DataSpace<simDim> guardingCells;
        for ( uint32_t d = 0; d < simDim; ++d )
            guardingCells[d] = ( relativMask[d] == -1 ? originGuard[d] : endGuard[d] );
        fieldE->addExchange( GUARD, i, guardingCells, FIELD_E );
    }
}

FieldE::~FieldE( )
{
    delete fieldE;
}

void FieldE::synchronize( )
{
    fieldE->deviceToHost( );
}

void FieldE::syncToDevice( )
{
    fieldE->hostToDevice( );
}

EventTask FieldE::asyncCommunication( EventTask serialEvent )
{
    return fieldE->asyncCommunication( serialEvent );
}

void FieldE::init( FieldB &fieldB, LaserPhysics &laserPhysics )
{
    this->fieldB = &fieldB;
    this->laser = &laserPhysics;

    DataConnector::getInstance( ).registerData( *this, FIELD_E );
}

FieldE::DataBoxType FieldE::getDeviceDataBox( )
{
    return fieldE->getDeviceBuffer( ).getDataBox( );
}

FieldE::DataBoxType FieldE::getHostDataBox( )
{
    return fieldE->getHostBuffer( ).getDataBox( );
}

GridBuffer<FieldE::ValueType, simDim> &FieldE::getGridBuffer( )
{
    return *fieldE;
}

GridLayout< simDim> FieldE::getGridLayout( )
{
    return cellDescription.getGridLayout( );
}

void FieldE::laserManipulation( uint32_t currentStep )
{
    VirtualWindow win=MovingWindow::getInstance().getVirtualWindow(currentStep);
    
    /* Disable laser if
     * - init time of laser is over or
     * - we have periodic boundaries in Y direction or
     * - we already performed a slide
     */
    if ( ( currentStep * DELTA_T ) >= laserProfile::INIT_TIME ||
         GridController<simDim>::getInstance( ).getCommunicationMask( ).isSet( TOP ) || win.slides != 0 ) return;

    DataSpace<simDim-1> gridBlocks;
    DataSpace<simDim-1> blockSize;
    gridBlocks.x()=fieldE->getGridLayout( ).getDataSpaceWithoutGuarding( ).x( ) / SuperCellSize::x;
    blockSize.x()=SuperCellSize::x;
#if(SIMDIM ==DIM3)
    gridBlocks.y()=fieldE->getGridLayout( ).getDataSpaceWithoutGuarding( ).z( ) / SuperCellSize::z;
    blockSize.y()=SuperCellSize::z;
#endif
    __cudaKernel( kernelLaserE )
        ( gridBlocks,
          blockSize )
        ( this->getDeviceDataBox( ), laser->getLaserManipulator( currentStep ) );
}

void FieldE::reset( uint32_t )
{
    fieldE->getHostBuffer( ).reset( true );
    fieldE->getDeviceBuffer( ).reset( false );
}

typename FieldE::UnitValueType
FieldE::getUnit( )
{
    return UnitValueType( UNIT_EFIELD, UNIT_EFIELD, UNIT_EFIELD );
}

std::string
FieldE::getName( )
{
    return "FieldE";
}

uint32_t
FieldE::getCommTag( )
{
    return FIELD_E;
}

}
