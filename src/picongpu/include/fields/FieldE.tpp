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

#include "math/Vector.hpp"

#include <list>

#include "particles/traits/GetInterpolation.hpp"
#include "particles/traits/FilterByFlag.hpp"
#include "traits/GetMargin.hpp"
#include "particles/traits/GetMarginPusher.hpp"
#include <boost/mpl/accumulate.hpp>
#include "fields/LaserPhysics.hpp"

namespace picongpu
{
using namespace PMacc;

FieldE::FieldE( MappingDesc cellDescription ) :
SimulationFieldHelper<MappingDesc>( cellDescription ),
fieldB( NULL )
{
    fieldE = new GridBuffer<ValueType, simDim > ( cellDescription.getGridLayout( ) );
    typedef typename PMacc::particles::traits::FilterByFlag
    <
        VectorAllSpecies,
        interpolation<>
    >::type VectorSpeciesWithInterpolation;

    typedef bmpl::accumulate<
        VectorSpeciesWithInterpolation,
        typename PMacc::math::CT::make_Int<simDim, 0>::type,
        PMacc::math::CT::max<bmpl::_1, GetLowerMargin< GetInterpolation<bmpl::_2> > >
        >::type LowerMarginInterpolation;

    typedef bmpl::accumulate<
        VectorSpeciesWithInterpolation,
        typename PMacc::math::CT::make_Int<simDim, 0>::type,
        PMacc::math::CT::max<bmpl::_1, GetUpperMargin< GetInterpolation<bmpl::_2> > >
        >::type UpperMarginInterpolation;

    /* Calculate the maximum Neighbors we need from MAX(ParticleShape, FieldSolver) */
    typedef PMacc::math::CT::max<
        LowerMarginInterpolation,
        GetMargin<fieldSolver::FieldSolver, FIELD_E>::LowerMargin
        >::type LowerMarginInterpolationAndSolver;
    typedef PMacc::math::CT::max<
        UpperMarginInterpolation,
        GetMargin<fieldSolver::FieldSolver, FIELD_E>::UpperMargin
        >::type UpperMarginInterpolationAndSolver;

    /* Calculate upper and lower margin for pusher 
       (currently all pusher use the interpolation of the species)  
       and find maximum margin 
    */
    typedef typename PMacc::particles::traits::FilterByFlag
    <
        VectorSpeciesWithInterpolation,
        particlePusher<>
    >::type VectorSpeciesWithPusherAndInterpolation;
    typedef bmpl::accumulate<
        VectorSpeciesWithPusherAndInterpolation,
        LowerMarginInterpolationAndSolver,
        PMacc::math::CT::max<bmpl::_1, GetLowerMarginPusher<bmpl::_2> >
        >::type LowerMargin;

    typedef bmpl::accumulate<
        VectorSpeciesWithPusherAndInterpolation,
        UpperMarginInterpolationAndSolver,
        PMacc::math::CT::max<bmpl::_1, GetUpperMarginPusher<bmpl::_2> >
        >::type UpperMargin;

    const DataSpace<simDim> originGuard( LowerMargin( ).toRT( ) );
    const DataSpace<simDim> endGuard( UpperMargin( ).toRT( ) );

    /*receive from all directions*/
    for ( uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i )
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
    __delete(fieldE);
}

SimulationDataId FieldE::getUniqueId()
{
    return getName();
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

    Environment<>::get().DataConnector().registerData( *this);
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
    const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);

    /* Disable laser if
     * - init time of laser is over or
     * - we have periodic boundaries in Y direction or
     * - we already performed a slide
     */
    if ( ( currentStep * DELTA_T ) >= laserProfile::INIT_TIME ||
         Environment<simDim>::get().GridController().getCommunicationMask( ).isSet( TOP ) || numSlides != 0 ) return;

    DataSpace<simDim-1> gridBlocks;
    DataSpace<simDim-1> blockSize;
    gridBlocks.x()=fieldE->getGridLayout( ).getDataSpaceWithoutGuarding( ).x( ) / SuperCellSize::x::value;
    blockSize.x()=SuperCellSize::x::value;
#if(SIMDIM ==DIM3)
    gridBlocks.y()=fieldE->getGridLayout( ).getDataSpaceWithoutGuarding( ).z( ) / SuperCellSize::z::value;
    blockSize.y()=SuperCellSize::z::value;
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


HDINLINE
FieldE::UnitValueType
FieldE::getUnit( )
{
    return UnitValueType( UNIT_EFIELD, UNIT_EFIELD, UNIT_EFIELD );
}

HDINLINE
std::vector<float_64>
FieldE::getUnitDimension( )
{
    /* L, M, T, I, theta, N, J
     *
     * E is in volts per meters: V / m = kg * m / (A * s^3)
     *   -> L * M * T^-3 * I^-1
     */
    std::vector<float_64> unitDimension( 7, 0.0 );
    unitDimension.at(0) =  1.0; // L^1
    unitDimension.at(1) =  1.0; // M^1
    unitDimension.at(2) = -3.0; // T^-3
    unitDimension.at(3) = -1.0; // I^-1

    return unitDimension;
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
