/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch, Felix Schmitt
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
#include "Particles.hpp"
#include <cassert>


#include "particles/Particles.kernel"

#include "dataManagement/DataConnector.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"
#include "fields/FieldTmp.hpp"

#include "particles/memory/buffers/ParticlesBuffer.hpp"
#include "ParticlesInit.kernel"
#include "mappings/simulation/GridController.hpp"
#include "mpi/SeedPerRank.hpp"

#include "simulationControl/MovingWindow.hpp"

#include <assert.h>
#include <limits>

#include "fields/numericalCellTypes/YeeCell.hpp"

#include "particles/traits/GetPusher.hpp"

namespace picongpu
{


using namespace PMacc;

template<typename T_ParticleDescription>
Particles<T_ParticleDescription>::Particles( GridLayout<simDim> gridLayout,
                                             MappingDesc cellDescription,
                                             SimulationDataId datasetID) :
ParticlesBase<T_ParticleDescription, MappingDesc>( cellDescription ),
fieldB( NULL ), fieldE( NULL ), fieldJurrent( NULL ), fieldTmp( NULL ), gridLayout( gridLayout ),
datasetID( datasetID )
{
    size_t sizeOfExchanges = 2 * 2 * ( BYTES_EXCHANGE_X + BYTES_EXCHANGE_Y + BYTES_EXCHANGE_Z ) + BYTES_EXCHANGE_X * 2 * 8;


    this->particlesBuffer = new BufferType( gridLayout.getDataSpace( ), gridLayout.getGuard( ) );

    log<picLog::MEMORY > ( "size for all exchange = %1% MiB" ) % ( (double) sizeOfExchanges / 1024. / 1024. );

    this->particlesBuffer->addExchange( Mask( LEFT ) + Mask( RIGHT ), BYTES_EXCHANGE_X, FrameType::CommunicationTag );
    this->particlesBuffer->addExchange( Mask( TOP ) + Mask( BOTTOM ), BYTES_EXCHANGE_Y, FrameType::CommunicationTag );
    //edges of the simulation area
    this->particlesBuffer->addExchange( Mask( RIGHT + TOP ) + Mask( LEFT + TOP ) +
                                        Mask( LEFT + BOTTOM ) + Mask( RIGHT + BOTTOM ), BYTES_EDGES, FrameType::CommunicationTag );

#if(SIMDIM==DIM3)
    this->particlesBuffer->addExchange( Mask( FRONT ) + Mask( BACK ), BYTES_EXCHANGE_Z, FrameType::CommunicationTag );
    //edges of the simulation area
    this->particlesBuffer->addExchange( Mask( FRONT + TOP ) + Mask( BACK + TOP ) +
                                        Mask( FRONT + BOTTOM ) + Mask( BACK + BOTTOM ),
                                        BYTES_EDGES, FrameType::CommunicationTag );
    this->particlesBuffer->addExchange( Mask( FRONT + RIGHT ) + Mask( BACK + RIGHT ) +
                                        Mask( FRONT + LEFT ) + Mask( BACK + LEFT ),
                                        BYTES_EDGES, FrameType::CommunicationTag );
    //corner of the simulation area
    this->particlesBuffer->addExchange( Mask( TOP + FRONT + RIGHT ) + Mask( TOP + BACK + RIGHT ) + Mask( BOTTOM + FRONT + RIGHT ) + Mask( BOTTOM + BACK + RIGHT ),
                                        BYTES_CORNER, FrameType::CommunicationTag );
    this->particlesBuffer->addExchange( Mask( TOP + FRONT + LEFT ) + Mask( TOP + BACK + LEFT ) + Mask( BOTTOM + FRONT + LEFT ) + Mask( BOTTOM + BACK + LEFT ),
                                        BYTES_CORNER, FrameType::CommunicationTag );
#endif
}

template< typename T_ParticleDescription>
void Particles<T_ParticleDescription>::createParticleBuffer( size_t gpuMemory )
{

    /*!\todo: this is the 4GB fix for GPUs with more than 4GB memory*/
    if ( gpuMemory >= UINT_MAX )
        gpuMemory = (size_t) ( UINT_MAX - 2 );

    this->particlesBuffer->createParticleBuffer( gpuMemory );

}

template< typename T_ParticleDescription>
Particles<T_ParticleDescription>::~Particles( )
{
    delete this->particlesBuffer;
}

template< typename T_ParticleDescription>
SimulationDataId Particles<T_ParticleDescription>::getUniqueId( )
{
    return datasetID;
}

template< typename T_ParticleDescription>
void Particles<T_ParticleDescription>::synchronize( )
{
    this->particlesBuffer->deviceToHost( );
}

template< typename T_ParticleDescription>
void Particles<T_ParticleDescription>::syncToDevice( )
{
    this->particlesBuffer->hostToDevice( );
}

template<typename T_ParticleDescription>
void Particles<T_ParticleDescription>::init( FieldE &fieldE, FieldB &fieldB, FieldJ &fieldJ, FieldTmp &fieldTmp )
{
    this->fieldE = &fieldE;
    this->fieldB = &fieldB;
    this->fieldJurrent = &fieldJ;
    this->fieldTmp = &fieldTmp;

    Environment<>::get( ).DataConnector().registerData( *this );
}

template<typename T_ParticleDescription>
void Particles<T_ParticleDescription>::update(uint32_t )
{
    typedef typename HasFlag<FrameType,particlePusher<> >::type hasPusher;
    typedef typename GetFlagType<FrameType,particlePusher<> >::type FoundPusher;

    /* if no pusher was defined we use PusherNone as fallback */
    typedef typename bmpl::if_<hasPusher,FoundPusher,particles::pusher::None >::type SelectPusher;
    typedef typename SelectPusher::type ParticlePush;

    typedef typename GetFlagType<FrameType,interpolation<> >::type::ThisType InterpolationScheme;

    typedef typename GetMargin<InterpolationScheme>::LowerMargin LowerMargin;
    typedef typename GetMargin<InterpolationScheme>::UpperMargin UpperMargin;

    typedef PushParticlePerFrame<ParticlePush, MappingDesc::SuperCellSize,
        InterpolationScheme,
        fieldSolver::NumericalCellType > FrameSolver;

    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        LowerMargin,
        UpperMargin
        > BlockArea;

    dim3 block( MappingDesc::SuperCellSize::toRT().toDim3() );

    __picKernelArea( kernelMoveAndMarkParticles<BlockArea>, this->cellDescription, CORE + BORDER )
        (block)
        ( this->getDeviceParticlesBox( ),
          this->fieldE->getDeviceDataBox( ),
          this->fieldB->getDeviceDataBox( ),
          FrameSolver( )
          );

    ParticlesBaseType::template shiftParticles < CORE + BORDER > ( );
}

template< typename T_ParticleDescription>
void Particles<T_ParticleDescription>::reset( uint32_t )
{
    this->particlesBuffer->reset( );
}

template< typename T_ParticleDescription>
void Particles<T_ParticleDescription>::initFill( uint32_t currentStep )
{
    Window window = MovingWindow::getInstance( ).getWindow( currentStep );
    const uint32_t numSlides = MovingWindow::getInstance( ).getSlideCounter( currentStep );
    PMACC_AUTO( simBox, Environment<simDim>::get().SubGrid().getSimulationBox( ) );

    /*calculate real simulation area offset from the beginning of the simulation*/
    DataSpace<simDim> localCells = gridLayout.getDataSpaceWithoutGuarding( );
    DataSpace<simDim> gpuCellOffset = simBox.getGlobalOffset( );
    gpuCellOffset.y( ) += numSlides * localCells.y( );

    GlobalSeed globalSeed;
    mpi::SeedPerRank<simDim> seedPerRank;
    uint32_t seed = seedPerRank( globalSeed(), FrameType::CommunicationTag );
    seed ^= POSITION_SEED;
    dim3 block( MappingDesc::SuperCellSize::toRT( ).toDim3() );

    if ( gasProfile::GAS_ENABLED )
    {
        const DataSpace<simDim> globalNrOfCells = simBox.getGlobalSize( );

        PMACC_AUTO( &fieldTmpGridBuffer, this->fieldTmp->getGridBuffer() );
        FieldTmp::DataBoxType dataBox = fieldTmpGridBuffer.getDeviceBuffer().getDataBox();

        if (!gasProfile::gasSetup(fieldTmpGridBuffer, window))
        {
            log<picLog::SIMULATION_STATE > ("Failed to setup gas profile");
        }

        __picKernelArea( kernelFillGridWithParticles, this->cellDescription, CORE + BORDER + GUARD )
            (block)
            ( this->particlesBuffer->getDeviceParticleBox( ),
              this->particlesBuffer->hasSendExchange( TOP ),
              gpuCellOffset,
              seed,
              globalNrOfCells.y( ),
              dataBox.shift(this->fieldTmp->getGridLayout().getGuard()));
    }

    this->fillAllGaps( );

    log<picLog::SIMULATION_STATE > ( "Wait for init particles finished (y offset = %1%)" ) % gpuCellOffset.y( );
    __getTransactionEvent( ).waitForFinished( );
}

template< typename T_ParticleDescription>
template< typename t_ParticleDescription>
void Particles<T_ParticleDescription>::deviceCloneFrom( Particles< t_ParticleDescription> &src )
{
    dim3 block( PMacc::math::CT::volume<SuperCellSize>::type::value );

    __picKernelArea( kernelCloneParticles, this->cellDescription, CORE + BORDER + GUARD )
        (block) ( this->getDeviceParticlesBox( ), src.getDeviceParticlesBox( ) );
    log<picLog::SIMULATION_STATE > ( "start clone particles" );
    this->fillAllGaps( );

    log<picLog::SIMULATION_STATE > ( "Wait for clone particles finished" );
    __getTransactionEvent( ).waitForFinished( );
}

template< typename T_ParticleDescription>
void Particles<T_ParticleDescription>::deviceAddTemperature( float_X energy )
{
    dim3 block( MappingDesc::SuperCellSize::toRT( ).toDim3() );

    GlobalSeed globalSeed;
    mpi::SeedPerRank<simDim> seedPerRank;
    uint32_t seed = seedPerRank( globalSeed(), FrameType::CommunicationTag );
    seed ^= TEMPERATURE_SEED;

    __picKernelArea( kernelAddTemperature, this->cellDescription, CORE + BORDER + GUARD )
        (block) ( this->getDeviceParticlesBox( ), energy, seed );

    log<picLog::SIMULATION_STATE > ( "Wait for addTemperature finished" );
    __getTransactionEvent( ).waitForFinished( );
}

template< typename T_ParticleDescription>
void Particles<T_ParticleDescription>::deviceSetDrift( uint32_t currentStep )
{
    const uint32_t numSlides = MovingWindow::getInstance( ).getSlideCounter( currentStep );

    dim3 block( MappingDesc::SuperCellSize::toRT( ).toDim3() );

    PMACC_AUTO( simBox, Environment<simDim>::get().SubGrid().getSimulationBox( ) );
    const DataSpace<simDim> localNrOfCells( simBox.getLocalSize( ) );
    const DataSpace<simDim> globalNrOfCells( simBox.getGlobalSize( ) );

    /* calculate real simulation area offset from the beginning of the simulation
     */
    uint32_t simulationYCell = simBox.getGlobalOffset( ).y( ) +
        ( numSlides * localNrOfCells.y( ) );

    __picKernelArea( kernelSetDrift, this->cellDescription, CORE + BORDER + GUARD )
        (block)
        ( this->particlesBuffer->getDeviceParticleBox( ),
          simulationYCell,
          globalNrOfCells.y( ) );

    log<picLog::SIMULATION_STATE > ( "Wait for set drift finished" );
    __getTransactionEvent( ).waitForFinished( );
}

} // end namespace
