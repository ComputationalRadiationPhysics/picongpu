/* Copyright 2013-2020 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
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
#include "picongpu/fields/currentDeposition/Deposit.hpp"
#include "picongpu/particles/traits/GetCurrentSolver.hpp"
#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"

#include <pmacc/particles/memory/boxes/ParticlesBox.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/fields/tasks/FieldFactory.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/MakeUnique.hpp>
#include <pmacc/fields/operations/CopyGuardToExchange.hpp>
#include <pmacc/fields/operations/AddExchangeToBorder.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>

#include <boost/mpl/accumulate.hpp>

#include <iostream>
#include <memory>


namespace picongpu
{

using namespace pmacc;

FieldJ::FieldJ( MappingDesc const & cellDescription ) :
    SimulationFieldHelper<MappingDesc>( cellDescription ),
    buffer( cellDescription.getGridLayout( ) ),
    fieldJrecv( nullptr )
{
    const DataSpace<simDim> coreBorderSize = cellDescription.getGridLayout( ).getDataSpaceWithoutGuarding( );

    /* cell margins the current might spread to due to particle shapes */
    using AllSpeciesWithCurrent = typename pmacc::particles::traits::FilterByFlag<
        VectorAllSpecies,
        current<>
    >::type;

    using LowerMarginShapes = bmpl::accumulate<
        AllSpeciesWithCurrent,
        typename pmacc::math::CT::make_Int<simDim, 0>::type,
        pmacc::math::CT::max<bmpl::_1, GetLowerMargin< GetCurrentSolver<bmpl::_2> > >
        >::type;

    using UpperMarginShapes = bmpl::accumulate<
        AllSpeciesWithCurrent,
        typename pmacc::math::CT::make_Int<simDim, 0>::type,
        pmacc::math::CT::max<bmpl::_1, GetUpperMargin< GetCurrentSolver<bmpl::_2> > >
        >::type;

    /* margins are always positive, also for lower margins
     * additional current interpolations and current filters on FieldJ might
     * spread the dependencies on neighboring cells
     *   -> use max(shape,filter) */
    using LowerMargin = pmacc::math::CT::max<
        LowerMarginShapes,
        GetMargin<typename fields::Solver::CurrentInterpolation>::LowerMargin
        >::type;

    using UpperMargin = pmacc::math::CT::max<
        UpperMarginShapes,
        GetMargin<typename fields::Solver::CurrentInterpolation>::UpperMargin
        >::type;

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
        buffer.addExchangeBuffer( i, guardingCells, FIELD_J );
    }

    /* Receive border values in own guard for "receive" communication pattern - necessary for current interpolation/filter */
    const DataSpace<simDim> originRecvGuard( GetMargin<typename fields::Solver::CurrentInterpolation>::LowerMargin( ).toRT( ) );
    const DataSpace<simDim> endRecvGuard( GetMargin<typename fields::Solver::CurrentInterpolation>::UpperMargin( ).toRT( ) );
    if( originRecvGuard != DataSpace<simDim>::create(0) ||
        endRecvGuard != DataSpace<simDim>::create(0) )
    {
        fieldJrecv = pmacc::memory::makeUnique< GridBuffer<ValueType, simDim > >(
            buffer.getDeviceBuffer(),
            cellDescription.getGridLayout( )
        );

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

GridBuffer<FieldJ::ValueType, simDim> &FieldJ::getGridBuffer( )
{
    return buffer;
}

GridLayout<simDim> FieldJ::getGridLayout( )
{
    return cellDescription.getGridLayout( );
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

void FieldJ::reset( uint32_t )
{
}

void FieldJ::synchronize( )
{
    buffer.deviceToHost( );
}

SimulationDataId FieldJ::getUniqueId( )
{
    return getName( );
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

void FieldJ::assign( ValueType value )
{
    buffer.getDeviceBuffer( ).setValue( value );
    //fieldJ.reset(false);
}

template<uint32_t T_area, class T_Species>
void FieldJ::computeCurrent( T_Species & species, uint32_t )
{
#if BOOST_COMP_HIP && !defined(PIC_DISABLE_COMPUTE_CURRENT_THREAD_LIMITER)
    // wrong results more than 256 threads are used
    constexpr int workerMultiplier = 1;
#else
    /* tuning parameter to use more workers than cells in a supercell
    * valid domain: 1 <= workerMultiplier
    */
    constexpr int workerMultiplier = 2;
#endif

    using FrameType = typename T_Species::FrameType;
    typedef typename pmacc::traits::Resolve<
        typename GetFlagType<FrameType, current<> >::type
    >::type ParticleCurrentSolver;

    using FrameSolver = currentSolver::ComputePerFrame<ParticleCurrentSolver, Velocity, MappingDesc::SuperCellSize>;

    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        typename GetMargin<ParticleCurrentSolver>::LowerMargin,
        typename GetMargin<ParticleCurrentSolver>::UpperMargin
    > BlockArea;

    constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
        pmacc::math::CT::volume< SuperCellSize >::type::value * workerMultiplier
    >::value;

    using Strategy = currentSolver::traits::GetStrategy_t< FrameSolver> ;

    auto const depositionKernel = currentSolver::KernelComputeCurrent<
        numWorkers,
        BlockArea
    >{};

    typename T_Species::ParticlesBoxType pBox = species.getDeviceParticlesBox( );
    FieldJ::DataBoxType jBox = buffer.getDeviceBuffer( ).getDataBox( );
    FrameSolver solver( DELTA_T );

    auto const deposit = currentSolver::Deposit< Strategy >{};
    deposit.template execute<
        T_area,
        numWorkers
    >(
        cellDescription,
        depositionKernel,
        solver,
        jBox,
        pBox
    );
}

template<uint32_t T_area, class T_CurrentInterpolation>
void FieldJ::addCurrentToEMF( T_CurrentInterpolation& myCurrentInterpolation )
{
    DataConnector &dc = Environment<>::get().DataConnector();
    auto fieldE = dc.get< FieldE >( FieldE::getName(), true );
    auto fieldB = dc.get< FieldB >( FieldB::getName(), true );

    AreaMapping<
        T_area,
        MappingDesc
    > mapper(cellDescription);

    constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
        pmacc::math::CT::volume< SuperCellSize >::type::value
    >::value;

    PMACC_KERNEL( currentSolver::KernelAddCurrentToEMF< numWorkers >{} )(
        mapper.getGridDim(),
        numWorkers
        )(
            fieldE->getDeviceDataBox( ),
            fieldB->getDeviceDataBox( ),
            buffer.getDeviceBuffer( ).getDataBox( ),
            myCurrentInterpolation,
            mapper
            );
    dc.releaseData( FieldE::getName() );
    dc.releaseData( FieldB::getName() );
}

void FieldJ::bashField( uint32_t exchangeType )
{
    pmacc::fields::operations::CopyGuardToExchange{ }(
        buffer,
        SuperCellSize{ },
        exchangeType
    );
}

void FieldJ::insertField( uint32_t exchangeType )
{
    pmacc::fields::operations::AddExchangeToBorder{ }(
        buffer,
        SuperCellSize{ },
        exchangeType
    );
}

} // namespace picongpu
