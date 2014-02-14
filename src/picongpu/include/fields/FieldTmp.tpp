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
 


#pragma once

#include "memory/buffers/GridBuffer.hpp"
#include "mappings/simulation/GridController.hpp"

#include "dataManagement/DataConnector.hpp"

#include "mappings/kernel/AreaMapping.hpp"
#include "eventSystem/EventSystem.hpp"
#include "mappings/kernel/ExchangeMapping.hpp"
#include "fields/tasks/FieldFactory.hpp"

#include "dimensions/SuperCellDescription.hpp"

#include "fields/FieldTmp.kernel"

#include "MaxwellSolver/Solvers.hpp"
#include "fields/numericalCellTypes/NumericalCellTypes.hpp"

#include "math/vector/compile-time/Vector.hpp"

namespace picongpu
{
    using namespace PMacc;

    FieldTmp::FieldTmp( MappingDesc cellDescription ) :
    SimulationFieldHelper<MappingDesc>( cellDescription )
    {
        fieldTmp = new GridBuffer<ValueType, simDim > ( cellDescription.getGridLayout( ) );

        /** \todo The exchange has to be resetted and set again regarding the
         *  temporary "Fill-"Functor we want to use.
         * 
         *  Problem: buffers don't allow "bigger" exchange during run time.
         *           so let's stay with the maximum guards.
         */

        const DataSpace<simDim> coreBorderSize = cellDescription.getGridLayout( ).getDataSpaceWithoutGuarding( );

        typedef picongpu::FieldToParticleInterpolationNative<
            ShiftCoordinateSystemNative,
            speciesParticleShape::ParticleShape::ChargeAssignment,
            AssignedTrilinearInterpolation> maxMarginsFrameSolver;

        /* The maximum Neighbors we need will be given by the ParticleShape */
        typedef typename GetMargin<maxMarginsFrameSolver>::LowerMargin LowerMargin;
        typedef typename GetMargin<maxMarginsFrameSolver>::UpperMargin UpperMargin;

        const DataSpace<simDim> originGuard( LowerMargin( ).vec( ) );
        const DataSpace<simDim> endGuard( UpperMargin( ).vec( ) );

        /*go over all directions*/
        for( uint32_t i = 1; i < numberOfNeighbors[simDim]; ++i )
        {
            DataSpace<simDim> relativMask = Mask::getRelativeDirections<simDim > ( i );
            /*guarding cells depend on direction
             */
            DataSpace<simDim> guardingCells;
            for( uint32_t d = 0; d < simDim; ++d )
            {
                /*originGuard and endGuard are switch because we send data 
                 * e.g. from left I get endGuardingCells and from right I originGuardingCells
                 */
                switch( relativMask[d] )
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
            fieldTmp->addExchangeBuffer( i, guardingCells, FIELD_TMP );
        }
    }

    FieldTmp::~FieldTmp( )
    {
        delete fieldTmp;
    }

    template<uint32_t AREA, class FrameSolver, class ParticlesClass>
    void FieldTmp::computeValue( ParticlesClass& parClass, uint32_t )
    {
        typedef SuperCellDescription<
            typename MappingDesc::SuperCellSize,
            typename toTVec<typename FrameSolver::LowerMargin>::type,
            typename toTVec<typename FrameSolver::UpperMargin>::type
            > BlockArea;

        StrideMapping<AREA, simDim, MappingDesc> mapper( cellDescription );
        typename ParticlesClass::ParticlesBoxType pBox = parClass.getDeviceParticlesBox( );
        FieldTmp::DataBoxType tmpBox = this->fieldTmp->getDeviceBuffer( ).getDataBox( );
        FrameSolver solver;

        __startAtomicTransaction( __getTransactionEvent( ) );
        do
        {
            __cudaKernel( ( kernelComputeSupercells<BlockArea, AREA> ) )
                ( mapper.getGridDim( ), mapper.getSuperCellSize( ) )
                ( tmpBox,
                  pBox, solver, mapper );
        } while( mapper.next( ) );
        __setTransactionEvent( __endTransaction( ) );
    }

    void FieldTmp::synchronize( )
    {
        fieldTmp->deviceToHost( );
    }

    void FieldTmp::syncToDevice( )
    {
        fieldTmp->hostToDevice( );
    }

    EventTask FieldTmp::asyncCommunication( EventTask serialEvent )
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

    void FieldTmp::bashField( uint32_t exchangeType )
    {
        ExchangeMapping<GUARD, MappingDesc> mapper( this->cellDescription, exchangeType );

        dim3 grid = mapper.getGridDim( );

        const DataSpace<simDim> direction = Mask::getRelativeDirections<simDim > ( mapper.getExchangeType( ) );
        __cudaKernel( kernelBashValue )
            ( grid, mapper.getSuperCellSize( ) )
            ( fieldTmp->getDeviceBuffer( ).getDataBox( ),
              fieldTmp->getSendExchange( exchangeType ).getDeviceBuffer( ).getDataBox( ),
              fieldTmp->getSendExchange( exchangeType ).getDeviceBuffer( ).getDataSpace( ),
              direction,
              mapper );
    }

    void FieldTmp::insertField( uint32_t exchangeType )
    {
        ExchangeMapping<GUARD, MappingDesc> mapper( this->cellDescription, exchangeType );

        dim3 grid = mapper.getGridDim( );

        const DataSpace<simDim> direction = Mask::getRelativeDirections<simDim > ( mapper.getExchangeType( ) );
        __cudaKernel( kernelInsertValue )
            ( grid, mapper.getSuperCellSize( ) )
            ( fieldTmp->getDeviceBuffer( ).getDataBox( ),
              fieldTmp->getReceiveExchange( exchangeType ).getDeviceBuffer( ).getDataBox( ),
              fieldTmp->getReceiveExchange( exchangeType ).getDeviceBuffer( ).getDataSpace( ),
              direction, mapper );
    }

    void FieldTmp::init( )
    {
        DataConnector::getInstance( ).registerData( *this, FIELD_TMP );
    }

    FieldTmp::DataBoxType FieldTmp::getDeviceDataBox( )
    {
        return fieldTmp->getDeviceBuffer( ).getDataBox( );
    }

    FieldTmp::DataBoxType FieldTmp::getHostDataBox( )
    {
        return fieldTmp->getHostBuffer( ).getDataBox( );
    }

    GridBuffer<FieldTmp::ValueType, simDim> &FieldTmp::getGridBuffer( )
    {
        return *fieldTmp;
    }

    GridLayout< simDim> FieldTmp::getGridLayout( )
    {
        return cellDescription.getGridLayout( );
    }

    void FieldTmp::reset( uint32_t )
    {
        fieldTmp->getHostBuffer( ).reset( true );
        fieldTmp->getDeviceBuffer( ).reset( false );
    }
    
    template<class FrameSolver >
    typename FieldTmp::UnitValueType
    FieldTmp::getUnit( )
    {
        return FrameSolver().getUnit();
    }
    

    template<class FrameSolver >
    std::string
    FieldTmp::getName( )
    {
        return FrameSolver().getName();
    }
    
    uint32_t
    FieldTmp::getCommTag( )
    {
        return FIELD_TMP;
    }

} // namespace picongpu

