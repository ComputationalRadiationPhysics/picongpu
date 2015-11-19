/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch, Benjamin Worpitz
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

#include <iostream>
#include "simulation_defines.hpp"
#include "FieldJ.hpp"
#include "fields/FieldJ.kernel"


#include "particles/memory/boxes/ParticlesBox.hpp"

#include "dataManagement/DataConnector.hpp"

#include "mappings/kernel/AreaMapping.hpp"
#include "mappings/kernel/StrideMapping.hpp"
#include "mappings/kernel/ExchangeMapping.hpp"
#include "fields/tasks/FieldFactory.hpp"

#include "fields/numericalCellTypes/NumericalCellTypes.hpp"

#include "math/Vector.hpp"

#include <boost/mpl/accumulate.hpp>
#include "particles/traits/GetCurrentSolver.hpp"
#include "traits/GetMargin.hpp"
#include "traits/Resolve.hpp"


namespace picongpu
{

using namespace PMacc;

FieldJ::FieldJ( MappingDesc cellDescription ) :
SimulationFieldHelper<MappingDesc>( cellDescription ),
fieldJ( cellDescription.getGridLayout( ) ), fieldE( NULL ), fieldB( NULL ), fieldJrecv( NULL )
{
    const DataSpace<simDim> coreBorderSize = cellDescription.getGridLayout( ).getDataSpaceWithoutGuarding( );

    /* cell margins the current might spread to due to particle shapes */
    typedef bmpl::accumulate<
        VectorAllSpecies,
        typename PMacc::math::CT::make_Int<simDim, 0>::type,
        PMacc::math::CT::max<bmpl::_1, GetLowerMargin< GetCurrentSolver<bmpl::_2> > >
        >::type LowerMarginShapes;

    typedef bmpl::accumulate<
        VectorAllSpecies,
        typename PMacc::math::CT::make_Int<simDim, 0>::type,
        PMacc::math::CT::max<bmpl::_1, GetUpperMargin< GetCurrentSolver<bmpl::_2> > >
        >::type UpperMarginShapes;

    /* margins are always positive, also for lower margins
     * additional current interpolations and current filters on FieldJ might
     * spread the dependencies on neighboring cells
     *   -> use max(shape,filter) */
    typedef PMacc::math::CT::max<
        LowerMarginShapes,
        GetMargin<fieldSolver::CurrentInterpolation>::LowerMargin
        >::type LowerMargin;

    typedef PMacc::math::CT::max<
        UpperMarginShapes,
        GetMargin<fieldSolver::CurrentInterpolation>::UpperMargin
        >::type UpperMargin;

    const DataSpace<simDim> originGuard( LowerMargin( ).toRT( ) );
    const DataSpace<simDim> endGuard( UpperMargin( ).toRT( ) );

    /*go over all directions*/
    for ( uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i )
    {
        DataSpace<simDim> relativMask = Mask::getRelativeDirections<simDim > ( i );
        /*guarding cells depend on direction
         */
        DataSpace<simDim> guardingCells;
        for ( uint32_t d = 0; d < simDim; ++d )
        {
            /*originGuard and endGuard are switch because we send data
             * e.g. from left I get endGuardingCells and from right I originGuardingCells
             */
            switch ( relativMask[d] )
            {
                /*receive from negativ side positiv (end) garding cells*/
            case -1: guardingCells[d] = endGuard[d];
                break;
                /*receive from positiv side negativ (origin) garding cells*/
            case 1: guardingCells[d] = originGuard[d];
                break;
            case 0: guardingCells[d] = coreBorderSize[d];
                break;
            };

        }
        // std::cout << "ex " << i << " x=" << guardingCells[0] << " y=" << guardingCells[1] << " z=" << guardingCells[2] << std::endl;
        fieldJ.addExchangeBuffer( i, guardingCells, FIELD_J );
    }

    /* Receive border values in own guard for "receive" communication pattern - necessary for current interpolation/filter */
    const DataSpace<simDim> originRecvGuard( GetMargin<fieldSolver::CurrentInterpolation>::LowerMargin( ).toRT( ) );
    const DataSpace<simDim> endRecvGuard( GetMargin<fieldSolver::CurrentInterpolation>::UpperMargin( ).toRT( ) );
    if( originRecvGuard != DataSpace<simDim>::create(0) ||
        endRecvGuard != DataSpace<simDim>::create(0) )
    {
        fieldJrecv = new GridBuffer<ValueType, simDim > ( fieldJ.getDeviceBuffer(), cellDescription.getGridLayout( ) );

        /*go over all directions*/
        for ( uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i )
        {
            DataSpace<simDim> relativMask = Mask::getRelativeDirections<simDim > ( i );
            /* guarding cells depend on direction
             * for negative direction use originGuard else endGuard (relative direction ZERO is ignored)
             * don't switch end and origin because this is a read buffer and no send buffer
             */
            DataSpace<simDim> guardingCells;
            for ( uint32_t d = 0; d < simDim; ++d )
                guardingCells[d] = ( relativMask[d] == -1 ? originRecvGuard[d] : endRecvGuard[d] );
            fieldJrecv->addExchange( GUARD, i, guardingCells, FIELD_JRECV );
        }
    }
}

FieldJ::~FieldJ( )
{
    __delete(fieldJrecv);
}

SimulationDataId FieldJ::getUniqueId( )
{
    return getName( );
}

void FieldJ::synchronize( )
{
    fieldJ.deviceToHost( );
}

GridBuffer<FieldJ::ValueType, simDim> &FieldJ::getGridBuffer( )
{
    return fieldJ;
}

EventTask FieldJ::asyncCommunication( EventTask serialEvent )
{
    EventTask ret;
    __startTransaction( serialEvent );
    FieldFactory::getInstance( ).createTaskFieldReceiveAndInsert( *this );
    ret = __endTransaction( );

    __startTransaction( serialEvent );
    FieldFactory::getInstance( ).createTaskFieldSend( *this );
    ret += __endTransaction( );

    if( fieldJrecv != NULL )
    {
        EventTask eJ = fieldJrecv->asyncCommunication( ret );
        return eJ;
    }
    else
        return ret;
}

void FieldJ::bashField( uint32_t exchangeType )
{
    ExchangeMapping<GUARD, MappingDesc> mapper( this->cellDescription, exchangeType );

    dim3 grid = mapper.getGridDim( );

    const DataSpace<simDim> direction = Mask::getRelativeDirections<simDim > ( mapper.getExchangeType( ) );
    __cudaKernel( kernelBashCurrent )
        ( grid, mapper.getSuperCellSize( ) )
        ( fieldJ.getDeviceBuffer( ).getDataBox( ),
          fieldJ.getSendExchange( exchangeType ).getDeviceBuffer( ).getDataBox( ),
          fieldJ.getSendExchange( exchangeType ).getDeviceBuffer( ).getDataSpace( ),
          direction,
          mapper );
}

void FieldJ::insertField( uint32_t exchangeType )
{
    ExchangeMapping<GUARD, MappingDesc> mapper( this->cellDescription, exchangeType );

    dim3 grid = mapper.getGridDim( );

    const DataSpace<simDim> direction = Mask::getRelativeDirections<simDim > ( mapper.getExchangeType( ) );
    __cudaKernel( kernelInsertCurrent )
        ( grid, mapper.getSuperCellSize( ) )
        ( fieldJ.getDeviceBuffer( ).getDataBox( ),
          fieldJ.getReceiveExchange( exchangeType ).getDeviceBuffer( ).getDataBox( ),
          fieldJ.getReceiveExchange( exchangeType ).getDeviceBuffer( ).getDataSpace( ),
          direction, mapper );
}

void FieldJ::init( FieldE &fieldE, FieldB &fieldB )
{
    this->fieldE = &fieldE;
    this->fieldB = &fieldB;

    Environment<>::get( ).DataConnector( ).registerData( *this );
}

GridLayout<simDim> FieldJ::getGridLayout( )
{
    return cellDescription.getGridLayout( );
}

void FieldJ::reset( uint32_t )
{
}

void FieldJ::assign( ValueType value )
{
    fieldJ.getDeviceBuffer( ).setValue( value );
    //fieldJ.reset(false);
}

HDINLINE
FieldJ::UnitValueType
FieldJ::getUnit( )
{
    const float_64 UNIT_CURRENT = UNIT_CHARGE / UNIT_TIME / ( UNIT_LENGTH * UNIT_LENGTH );
    return UnitValueType( UNIT_CURRENT, UNIT_CURRENT, UNIT_CURRENT );
}

HDINLINE
std::vector<float_64>
FieldJ::getUnitDimension( )
{
    /* L, M, T, I, theta, N, J
     *
     * J is in A/m^2
     *   -> L^-2 * I
     */
    std::vector<float_64> unitDimension( 7, 0.0 );
    unitDimension.at(0) = -2.0; // L^-2
    unitDimension.at(3) =  1.0; // I^1

    return unitDimension;
}

std::string
FieldJ::getName( )
{
    return "FieldJ";
}

uint32_t
FieldJ::getCommTag( )
{
    return FIELD_J;
}

template<uint32_t AREA, class ParticlesClass>
void FieldJ::computeCurrent( ParticlesClass &parClass, uint32_t )
{
    /** tune paramter to use more threads than cells in a supercell
     *  valid domain: 1 <= workerMultiplier
     */
    const int workerMultiplier = 2;

    typedef typename ParticlesClass::FrameType FrameType;
    typedef typename PMacc::traits::Resolve<
        typename GetFlagType<FrameType, current<> >::type
    >::type ParticleCurrentSolver;

    typedef ComputeCurrentPerFrame<ParticleCurrentSolver, Velocity, MappingDesc::SuperCellSize> FrameSolver;

    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        typename GetMargin<ParticleCurrentSolver>::LowerMargin,
        typename GetMargin<ParticleCurrentSolver>::UpperMargin
        > BlockArea;

    StrideMapping<AREA, 3, MappingDesc> mapper( cellDescription );
    typename ParticlesClass::ParticlesBoxType pBox = parClass.getDeviceParticlesBox( );
    FieldJ::DataBoxType jBox = this->fieldJ.getDeviceBuffer( ).getDataBox( );
    FrameSolver solver( DELTA_T );

    DataSpace<simDim> blockSize( mapper.getSuperCellSize( ) );
    blockSize[simDim - 1] *= workerMultiplier;

    do
    {
        __cudaKernel( ( kernelComputeCurrent<workerMultiplier, BlockArea, AREA> ) )
            ( mapper.getGridDim( ), blockSize )
            ( jBox,
              pBox, solver, mapper );
    }
    while ( mapper.next( ) );

}

template<uint32_t AREA, class T_CurrentInterpolation>
void FieldJ::addCurrentToEMF( T_CurrentInterpolation& myCurrentInterpolation )
{
    __picKernelArea( ( kernelAddCurrentToEMF ),
                     cellDescription,
                     AREA )
        ( MappingDesc::SuperCellSize::toRT( ).toDim3( ) )
        ( this->fieldE->getDeviceDataBox( ),
          this->fieldB->getDeviceDataBox( ),
          this->fieldJ.getDeviceBuffer( ).getDataBox( ),
          myCurrentInterpolation );
}

}
