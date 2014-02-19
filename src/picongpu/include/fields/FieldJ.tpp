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
#include "FieldJ.hpp"
#include "fields/FieldJ.kernel"


#include "particles/memory/boxes/ParticlesBox.hpp"

#include "dataManagement/DataConnector.hpp"

#include "mappings/kernel/AreaMapping.hpp"
#include "mappings/kernel/StrideMapping.hpp"
#include "mappings/kernel/ExchangeMapping.hpp"
#include "fields/tasks/FieldFactory.hpp"

#include "fields/numericalCellTypes/NumericalCellTypes.hpp"

#include "math/vector/compile-time/Vector.hpp"

namespace picongpu
{

using namespace PMacc;

FieldJ::FieldJ( MappingDesc cellDescription ) :
SimulationFieldHelper<MappingDesc>( cellDescription ),
fieldJ( cellDescription.getGridLayout( ) ), fieldE( NULL )
{
    typedef currentSolver::CurrentSolver ParticleCurrentSolver;

    const DataSpace<simDim> coreBorderSize = cellDescription.getGridLayout( ).getDataSpaceWithoutGuarding( );

    typedef typename GetMargin<ParticleCurrentSolver>::LowerMargin LowerMargin;
    typedef typename GetMargin<ParticleCurrentSolver>::UpperMargin UpperMargin;

    const DataSpace<simDim> originGuard( LowerMargin( ).vec( ) );
    const DataSpace<simDim> endGuard( UpperMargin( ).vec( ) );

    /*go over all directions*/
    for ( uint32_t i = 1; i < numberOfNeighbors[simDim]; ++i )
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
}

FieldJ::~FieldJ( )
{
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

void FieldJ::init( FieldE &fieldE )
{
    this->fieldE = &fieldE;

    DataConnector::getInstance( ).registerData( *this, FIELD_J );
}

GridLayout<simDim> FieldJ::getGridLayout( )
{
    return cellDescription.getGridLayout( );
}

void FieldJ::reset( uint32_t )
{
}

void FieldJ::clear( )
{
    ValueType tmp = float3_X( 0., 0., 0. );
    fieldJ.getDeviceBuffer( ).setValue( tmp );
    //fieldJ.reset(false);
}

typename FieldJ::UnitValueType
FieldJ::getUnit( )
{
    const UnitValueType unitaryVector( 1.0, 1.0, 1.0 );
    return unitaryVector * UNIT_CHARGE / UNIT_TIME / (UNIT_LENGTH * UNIT_LENGTH);
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
void FieldJ::computeCurrent( ParticlesClass &parClass, uint32_t ) throw (std::invalid_argument )
{
    /** tune paramter to use more threads than cells in a supercell
     *  valid domain: 1 <= workerMultiplier
     */
    const int workerMultiplier =2;
    
    typedef currentSolver::CurrentSolver ParticleCurrentSolver;
    typedef ComputeCurrentPerFrame<ParticleCurrentSolver, Velocity, MappingDesc::SuperCellSize> FrameSolver;

    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        typename toTVec<GetMargin<currentSolver::CurrentSolver>::LowerMargin>::type,
        typename toTVec<GetMargin<currentSolver::CurrentSolver>::UpperMargin>::type
        > BlockArea;

    StrideMapping<AREA, simDim, MappingDesc> mapper( cellDescription );
    typename ParticlesClass::ParticlesBoxType pBox = parClass.getDeviceParticlesBox( );
    FieldJ::DataBoxType jBox = this->fieldJ.getDeviceBuffer( ).getDataBox( );
    floatD_X cellSize;
    for(uint32_t i=0;i<simDim;++i)
        cellSize[i]=cell_size[i];
    FrameSolver solver(
                        cellSize,
                        DELTA_T );
    
    DataSpace<simDim> blockSize(mapper.getSuperCellSize( ));
    blockSize[simDim-1]*=workerMultiplier;

    __startAtomicTransaction( __getTransactionEvent( ) );
    do
    {
        __cudaKernel( ( kernelComputeCurrent<workerMultiplier,BlockArea, AREA> ) )
            ( mapper.getGridDim( ), blockSize )
            ( jBox,
              pBox, solver, mapper );
    }
    while ( mapper.next( ) );
    __setTransactionEvent( __endTransaction( ) );
}

template<uint32_t AREA>
void FieldJ::addCurrentToE( )
{
    __picKernelArea( ( kernelAddCurrentToE ),
                     cellDescription,
                     AREA )
        ( MappingDesc::SuperCellSize::getDataSpace( ) )
        ( this->fieldE->getDeviceDataBox( ),
          this->fieldJ.getDeviceBuffer( ).getDataBox( ) );
}

}
