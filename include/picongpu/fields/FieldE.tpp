/* Copyright 2013-2018 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
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

#include "picongpu/fields/FieldE.kernel"

#include "MaxwellSolver/Solvers.hpp"
#include "picongpu/fields/numericalCellTypes/NumericalCellTypes.hpp"

#include <pmacc/math/Vector.hpp>

#include "picongpu/particles/traits/GetInterpolation.hpp"
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"
#include "picongpu/particles/traits/GetMarginPusher.hpp"
#include "picongpu/fields/LaserPhysics.hpp"

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
        GetMargin<fieldSolver::FieldSolver, FIELD_E>::LowerMargin
        >::type LowerMarginInterpolationAndSolver;
    typedef pmacc::math::CT::max<
        UpperMarginInterpolation,
        GetMargin<fieldSolver::FieldSolver, FIELD_E>::UpperMargin
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

void FieldE::laserManipulation( uint32_t currentStep )
{
    /* initialize the laser not in the first cell is equal to a negative shift
     * in time
     */
    constexpr float_X laserTimeShift = laser::initPlaneY * CELL_HEIGHT / SPEED_OF_LIGHT;

    const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);

    /* Disable laser if
     * - init time of laser is over or
     * - we have periodic boundaries in Y direction or
     * - we already performed a slide
     */
    bool const laserNone = ( laserProfile::INIT_TIME == float_X(0.0) );
    bool const laserInitTimeOver =
        ( ( currentStep * DELTA_T  - laserTimeShift ) >= laserProfile::INIT_TIME );
    bool const topBoundariesArePeriodic =
        ( Environment<simDim>::get().GridController().getCommunicationMask( ).isSet( TOP ) );
    bool const boxHasSlided = ( numSlides != 0 );

    bool const disableLaser =
        laserNone ||
        laserInitTimeOver ||
        topBoundariesArePeriodic ||
        boxHasSlided;
    if( !disableLaser )
    {
        PMACC_VERIFY_MSG(
            laser::initPlaneY < static_cast<uint32_t>( Environment<simDim>::get().SubGrid().getLocalDomain().size.y() ),
            "initPlaneY must be located in the top GPU"
        );

        PMACC_CASSERT_MSG(
            __initPlaneY_needs_to_be_greate_than_the_top_absorber_cells_or_zero,
            laser::initPlaneY > ABSORBER_CELLS[1][0] ||
            laser::initPlaneY == 0 ||
            laserProfile::INIT_TIME == float_X(0.0) /* laser is disabled e.g. laserNone */
        );

        /* Calculate how many neighbors to the left we have
         * to initialize the laser in the E-Field
         *
         * Example: Yee needs one neighbor to perform dB = curlE
         *            -> initialize in y=0 plane
         *          A second order solver could need 2 neighbors left:
         *            -> initialize in y=0 and y=1 plane
         *
         * Question: Why do other codes initialize the B-Field instead?
         * Answer:   Because our fields are defined on the lower cell side
         *           (C-Style ftw). Therefore, our curls (for example Yee)
         *           are shifted nabla+ <-> nabla- compared to Fortran codes
         *           (in other words: curlLeft <-> curlRight)
         *           for E and B.
         *           For this reason, we have to initialize E instead of B.
         *
         * Problem: that's still not our case. For example our Yee does a
         *          dE = curlLeft(B) - therefor, we should init B, too.
         *
         *
         *  @todo: might also lack temporal offset since our formulas are E(x,z,t) instead of E(x,y,z,t)
         *  `const int max_y_neighbors = Get<fieldSolver::FieldSolver::OffsetOrigin_E, 1 >::value;`
         *
         * @todo Right now, the phase could be wrong ( == is cloned)
         *       @see LaserPhysics.hpp
         *
         * @todo What about the B-Field in the second plane?
         *
         */
        constexpr int laserInitCellsInY = 1;

        using LaserPlaneSizeInSuperCells = typename pmacc::math::CT::AssignIfInRange<
                typename SuperCellSize::vector_type,
                bmpl::integral_c< uint32_t, 1 >, /* y direction */
                bmpl::integral_c< int, laserInitCellsInY >
        >::type;

        DataSpace<simDim> gridBlocks = fieldE->getGridLayout( ).getDataSpaceWithoutGuarding( ) / SuperCellSize::toRT();
        // use the one supercell in y to initialize the laser plane
        gridBlocks.y() = 1;

        constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
            pmacc::math::CT::volume< LaserPlaneSizeInSuperCells >::type::value
        >::value;

        LaserPhysics laser( fieldE->getGridLayout() );

        PMACC_KERNEL(
            KernelLaserE<
                numWorkers,
                LaserPlaneSizeInSuperCells
            >{}
        )(
            gridBlocks,
            numWorkers
        )(
            this->getDeviceDataBox( ),
            laser.getLaserManipulator( currentStep )
        );
    }
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
