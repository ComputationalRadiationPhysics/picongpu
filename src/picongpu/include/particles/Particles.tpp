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

// Ionization
#include "particles/ionization/ionization.hpp"

#include "dataManagement/DataConnector.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"
#include "fields/FieldTmp.hpp"

#include "particles/memory/buffers/ParticlesBuffer.hpp"
#include "ParticlesInit.kernel"
#include "mappings/simulation/GridController.hpp"

#include "simulationControl/MovingWindow.hpp"

#include <assert.h>
#include <limits>

#include "fields/numericalCellTypes/YeeCell.hpp"

#include "traits/Resolve.hpp"

namespace picongpu
{


using namespace PMacc;

template<typename T_ParticleDescription>
Particles<T_ParticleDescription>::Particles( GridLayout<simDim> gridLayout,
                                             MappingDesc cellDescription,
                                             SimulationDataId datasetID ) :
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

}

template< typename T_ParticleDescription>
void Particles<T_ParticleDescription>::syncToDevice( )
{

}

template<typename T_ParticleDescription>
void Particles<T_ParticleDescription>::init( FieldE &fieldE, FieldB &fieldB, FieldJ &fieldJ, FieldTmp &fieldTmp )
{
    this->fieldE = &fieldE;
    this->fieldB = &fieldB;
    this->fieldJurrent = &fieldJ;
    this->fieldTmp = &fieldTmp;

    Environment<>::get( ).DataConnector( ).registerData( *this );
}

template<typename T_ParticleDescription>
void Particles<T_ParticleDescription>::update( uint32_t )
{
    typedef typename HasFlag<FrameType, particlePusher<> >::type hasPusher;
    typedef typename GetFlagType<FrameType, particlePusher<> >::type FoundPusher;

    /* if no pusher was defined we use PusherNone as fallback */
    typedef typename bmpl::if_<hasPusher, FoundPusher, particles::pusher::None >::type SelectPusher;
    typedef typename PMacc::traits::Resolve<SelectPusher>::type::type ParticlePush;

    typedef typename PMacc::traits::Resolve<
        typename GetFlagType<FrameType, interpolation<> >::type
        >::type InterpolationScheme;

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

    dim3 block( MappingDesc::SuperCellSize::toRT( ).toDim3( ) );

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
template<typename T_GasFunctor, typename T_PositionFunctor>
void Particles<T_ParticleDescription>::initGas( T_GasFunctor& gasFunctor,
                                                T_PositionFunctor& positionFunctor,
                                                const uint32_t currentStep )
{
    log<picLog::SIMULATION_STATE > ( "start create gas for species %1%" ) % FrameType::getName( );

    const uint32_t numSlides = MovingWindow::getInstance( ).getSlideCounter( currentStep );
    const SubGrid<simDim>& subGrid = Environment<simDim>::get( ).SubGrid( );
    DataSpace<simDim> localCells = subGrid.getLocalDomain( ).size;
    DataSpace<simDim> totalGpuCellOffset = subGrid.getLocalDomain( ).offset;
    totalGpuCellOffset.y( ) += numSlides * localCells.y( );

    dim3 block( MappingDesc::SuperCellSize::toRT( ).toDim3( ) );
    __picKernelArea( kernelFillGridWithParticles, this->cellDescription, CORE + BORDER )
        (block)
        ( gasFunctor, positionFunctor, totalGpuCellOffset, this->particlesBuffer->getDeviceParticleBox( ) );


    this->fillAllGaps( );
}

template< typename T_ParticleDescription>
template< typename t_ParticleDescription>
void Particles<T_ParticleDescription>::deviceCloneFrom( Particles< t_ParticleDescription> &src )
{
    dim3 block( PMacc::math::CT::volume<SuperCellSize>::type::value );

    __picKernelArea( kernelCloneParticles, this->cellDescription, CORE + BORDER )
        (block) ( this->getDeviceParticlesBox( ), src.getDeviceParticlesBox( ) );
    log<picLog::SIMULATION_STATE > ( "start clone species %1%" ) % FrameType::getName( );
    this->fillAllGaps( );
}

template< typename T_ParticleDescription>
template< typename T_Functor>
void Particles<T_ParticleDescription>::manipulateAllParticles( uint32_t currentStep, T_Functor& functor )
{

    dim3 block( MappingDesc::SuperCellSize::toRT( ).toDim3( ) );

    __picKernelArea( kernelManipulateAllParticles, this->cellDescription, CORE + BORDER )
        (block)
        ( this->particlesBuffer->getDeviceParticleBox( ),
          functor );
}

} // end namespace
