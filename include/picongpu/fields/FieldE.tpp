/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
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

#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>

#include <pmacc/dataManagement/DataConnector.hpp>

#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/eventSystem/EventSystem.hpp>
#include <pmacc/mappings/kernel/ExchangeMapping.hpp>

#include "picongpu/fields/FieldManipulator.hpp"
#include <pmacc/dimensions/SuperCellDescription.hpp>

#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/fields/numericalCellTypes/NumericalCellTypes.hpp"

#include <pmacc/math/Vector.hpp>

#include "picongpu/particles/traits/GetInterpolation.hpp"
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"
#include "picongpu/particles/traits/GetMarginPusher.hpp"

#include <boost/mpl/accumulate.hpp>

#include <list>
#include <memory>


namespace picongpu
{
using namespace pmacc;

FieldE::FieldE( MappingDesc cellDescription ) :
SimulationFieldHelper<MappingDesc>( cellDescription )
{
    fieldE = new GridBuffer<ValueType, simDim > ( cellDescription.getGridLayout( ) );
    typedef typename pmacc::particles::traits::FilterByFlag
    <
        VectorAllSpecies,
        interpolation<>
    >::type VectorSpeciesWithInterpolation;

    typedef bmpl::accumulate<
        VectorSpeciesWithInterpolation,
        typename pmacc::math::CT::make_Int<simDim, 0>::type,
        pmacc::math::CT::max<bmpl::_1, GetLowerMargin< GetInterpolation<bmpl::_2> > >
        >::type LowerMarginInterpolation;

    typedef bmpl::accumulate<
        VectorSpeciesWithInterpolation,
        typename pmacc::math::CT::make_Int<simDim, 0>::type,
        pmacc::math::CT::max<bmpl::_1, GetUpperMargin< GetInterpolation<bmpl::_2> > >
        >::type UpperMarginInterpolation;

    /* Calculate the maximum Neighbors we need from MAX(ParticleShape, FieldSolver) */
    typedef pmacc::math::CT::max<
        LowerMarginInterpolation,
        GetMargin<fields::Solver, FIELD_E>::LowerMargin
        >::type LowerMarginInterpolationAndSolver;
    typedef pmacc::math::CT::max<
        UpperMarginInterpolation,
        GetMargin<fields::Solver, FIELD_E>::UpperMargin
        >::type UpperMarginInterpolationAndSolver;

    /* Calculate upper and lower margin for pusher
       (currently all pusher use the interpolation of the species)
       and find maximum margin
    */
    typedef typename pmacc::particles::traits::FilterByFlag
    <
        VectorSpeciesWithInterpolation,
        particlePusher<>
    >::type VectorSpeciesWithPusherAndInterpolation;
    typedef bmpl::accumulate<
        VectorSpeciesWithPusherAndInterpolation,
        LowerMarginInterpolationAndSolver,
        pmacc::math::CT::max<bmpl::_1, GetLowerMarginPusher<bmpl::_2> >
        >::type LowerMargin;

    typedef bmpl::accumulate<
        VectorSpeciesWithPusherAndInterpolation,
        UpperMarginInterpolationAndSolver,
        pmacc::math::CT::max<bmpl::_1, GetUpperMarginPusher<bmpl::_2> >
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

HINLINE
std::vector<float_64>
FieldE::getUnitDimension( )
{
    /* L, M, T, I, theta, N, J
     *
     * E is in volts per meters: V / m = kg * m / (A * s^3)
     *   -> L * M * T^-3 * I^-1
     */
    std::vector<float_64> unitDimension( 7, 0.0 );
    unitDimension.at(SIBaseUnits::length) =  1.0;
    unitDimension.at(SIBaseUnits::mass)   =  1.0;
    unitDimension.at(SIBaseUnits::time)   = -3.0;
    unitDimension.at(SIBaseUnits::electricCurrent) = -1.0;

    return unitDimension;
}

std::string
FieldE::getName( )
{
    return "E";
}

uint32_t
FieldE::getCommTag( )
{
    return FIELD_E;
}

}
