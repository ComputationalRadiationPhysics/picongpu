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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/FieldJ.kernel"


#include <pmacc/particles/memory/boxes/ParticlesBox.hpp>

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/kernel/StrideMapping.hpp>
#include <pmacc/fields/tasks/FieldFactory.hpp>

#include "picongpu/fields/numericalCellTypes/NumericalCellTypes.hpp"

#include <pmacc/math/Vector.hpp>
#include <pmacc/fields/operations/CopyGuardToExchange.hpp>
#include <pmacc/fields/operations/AddExchangeToBorder.hpp>
#include "picongpu/particles/traits/GetCurrentSolver.hpp"
#include "picongpu/traits/GetMargin.hpp"
#include <pmacc/traits/Resolve.hpp>
#include "picongpu/traits/SIBaseUnits.hpp"
#include <pmacc/traits/GetNumWorkers.hpp>

#include <boost/mpl/accumulate.hpp>

#include <iostream>
#include <memory>


namespace picongpu
{

using namespace pmacc;

FieldJ::FieldJ( MappingDesc cellDescription ) :
SimulationFieldHelper<MappingDesc>( cellDescription ),
fieldJ( cellDescription.getGridLayout( ) ), fieldJrecv( nullptr )
{
    const DataSpace<simDim> coreBorderSize = cellDescription.getGridLayout( ).getDataSpaceWithoutGuarding( );

    /* cell margins the current might spread to due to particle shapes */
    typedef typename pmacc::particles::traits::FilterByFlag<
        VectorAllSpecies,
        current<>
    >::type AllSpeciesWithCurrent;

    typedef bmpl::accumulate<
        AllSpeciesWithCurrent,
        typename pmacc::math::CT::make_Int<simDim, 0>::type,
        pmacc::math::CT::max<bmpl::_1, GetLowerMargin< GetCurrentSolver<bmpl::_2> > >
        >::type LowerMarginShapes;

    typedef bmpl::accumulate<
        AllSpeciesWithCurrent,
        typename pmacc::math::CT::make_Int<simDim, 0>::type,
        pmacc::math::CT::max<bmpl::_1, GetUpperMargin< GetCurrentSolver<bmpl::_2> > >
        >::type UpperMarginShapes;

    /* margins are always positive, also for lower margins
     * additional current interpolations and current filters on FieldJ might
     * spread the dependencies on neighboring cells
     *   -> use max(shape,filter) */
    typedef pmacc::math::CT::max<
        LowerMarginShapes,
        GetMargin<typename fields::Solver::CurrentInterpolation>::LowerMargin
        >::type LowerMargin;

    typedef pmacc::math::CT::max<
        UpperMarginShapes,
        GetMargin<typename fields::Solver::CurrentInterpolation>::UpperMargin
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
                // receive from negativ side to positiv (end) guarding cells
            case -1: guardingCells[d] = endGuard[d];
                break;
                // receive from positiv side to negativ (origin) guarding cells
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
    const DataSpace<simDim> originRecvGuard( GetMargin<typename fields::Solver::CurrentInterpolation>::LowerMargin( ).toRT( ) );
    const DataSpace<simDim> endRecvGuard( GetMargin<typename fields::Solver::CurrentInterpolation>::UpperMargin( ).toRT( ) );
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

    if( fieldJrecv != nullptr )
    {
        EventTask eJ = fieldJrecv->asyncCommunication( ret );
        return eJ;
    }
    else
        return ret;
}

void FieldJ::bashField( uint32_t exchangeType )
{
    pmacc::fields::operations::CopyGuardToExchange{ }(
        fieldJ,
        SuperCellSize{ },
        exchangeType
    );
}

void FieldJ::insertField( uint32_t exchangeType )
{
    pmacc::fields::operations::AddExchangeToBorder{ }(
        fieldJ,
        SuperCellSize{ },
        exchangeType
    );
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

HINLINE
std::vector<float_64>
FieldJ::getUnitDimension( )
{
    /* L, M, T, I, theta, N, J
     *
     * J is in A/m^2
     *   -> L^-2 * I
     */
    std::vector<float_64> unitDimension( 7, 0.0 );
    unitDimension.at(SIBaseUnits::length) = -2.0;
    unitDimension.at(SIBaseUnits::electricCurrent) =  1.0;

    return unitDimension;
}

std::string
FieldJ::getName( )
{
    return "J";
}

uint32_t
FieldJ::getCommTag( )
{
    return FIELD_J;
}

template<uint32_t AREA, class ParticlesClass>
void FieldJ::computeCurrent( ParticlesClass &parClass, uint32_t )
{
    /* tuning parameter to use more workers than cells in a supercell
     * valid domain: 1 <= workerMultiplier
     */
    const int workerMultiplier = 2;

    using FrameType = typename ParticlesClass::FrameType;
    typedef typename pmacc::traits::Resolve<
        typename GetFlagType<FrameType, current<> >::type
    >::type ParticleCurrentSolver;

    typedef ComputeCurrentPerFrame<ParticleCurrentSolver, Velocity, MappingDesc::SuperCellSize> FrameSolver;

    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        typename GetMargin<ParticleCurrentSolver>::LowerMargin,
        typename GetMargin<ParticleCurrentSolver>::UpperMargin
    > BlockArea;

    /* The needed stride for the stride mapper depends on the stencil width.
     * If the upper and lower margin of the stencil fits into one supercell
     * a double checker board (stride 2) is needed.
     * The round up sum of margins is the number of supercells to skip.
     */
    using MarginPerDim = typename pmacc::math::CT::add<
        typename GetMargin<ParticleCurrentSolver>::LowerMargin,
        typename GetMargin<ParticleCurrentSolver>::UpperMargin
    >::type;
    using MaxMargin = typename pmacc::math::CT::max< MarginPerDim >::type;
    using SuperCellMinSize = typename pmacc::math::CT::min< SuperCellSize >::type;

    /* number of supercells which must be skipped to avoid overlapping areas
     * between different blocks in the kernel
     */
    constexpr uint32_t skipSuperCells = ( MaxMargin::value + SuperCellMinSize::value - 1u ) / SuperCellMinSize::value;
    StrideMapping<
        AREA,
        skipSuperCells + 1u, // stride 1u means each supercell is used
        MappingDesc
    > mapper( cellDescription );

    typename ParticlesClass::ParticlesBoxType pBox = parClass.getDeviceParticlesBox( );
    FieldJ::DataBoxType jBox = this->fieldJ.getDeviceBuffer( ).getDataBox( );
    FrameSolver solver( DELTA_T );

    constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
        pmacc::math::CT::volume< SuperCellSize >::type::value * workerMultiplier
    >::value;

    do
    {
        PMACC_KERNEL( KernelComputeCurrent< numWorkers, BlockArea >{} )
            ( mapper.getGridDim( ), numWorkers )
            ( jBox,
              pBox, solver, mapper );
    }
    while ( mapper.next( ) );

}

template<uint32_t AREA, class T_CurrentInterpolation>
void FieldJ::addCurrentToEMF( T_CurrentInterpolation& myCurrentInterpolation )
{
    DataConnector &dc = Environment<>::get().DataConnector();
    auto fieldE = dc.get< FieldE >( FieldE::getName(), true );
    auto fieldB = dc.get< FieldB >( FieldB::getName(), true );

    AreaMapping<
        AREA,
        MappingDesc
    > mapper(cellDescription);

    constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
        pmacc::math::CT::volume< SuperCellSize >::type::value
    >::value;

    PMACC_KERNEL( KernelAddCurrentToEMF< numWorkers >{} )(
        mapper.getGridDim(),
        numWorkers
    )(
        fieldE->getDeviceDataBox( ),
        fieldB->getDeviceDataBox( ),
        this->fieldJ.getDeviceBuffer( ).getDataBox( ),
        myCurrentInterpolation,
        mapper
    );
    dc.releaseData( FieldE::getName() );
    dc.releaseData( FieldB::getName() );
}

}
