/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
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
#include "Particles.hpp"
#include <cassert>


#include "particles/Particles.kernel"

#include "dataManagement/DataConnector.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"

#include "particles/memory/buffers/ParticlesBuffer.hpp"
#include "ParticlesInit.kernel"
#include "mappings/simulation/GridController.hpp"

#include "simulationControl/MovingWindow.hpp"

#include <assert.h>
#include <limits>

#include "fields/numericalCellTypes/YeeCell.hpp"

namespace picongpu
{


using namespace PMacc;

template< typename T_DataVector, typename T_MethodsVector>
Particles<T_DataVector, T_MethodsVector>::Particles( GridLayout<simDim> gridLayout,
                                                     MappingDesc cellDescription ) :
ParticlesBase<T_DataVector, T_MethodsVector, MappingDesc>( cellDescription ), fieldB( NULL ), fieldE( NULL ), fieldJurrent( NULL ), gridLayout( gridLayout )
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

template< typename T_DataVector, typename T_MethodsVector>
void Particles<T_DataVector, T_MethodsVector>::createParticleBuffer( size_t gpuMemory )
{

    /*!\todo: this is the 4GB fix for GPUs with more than 4GB memory*/
    if ( gpuMemory >= UINT_MAX )
        gpuMemory = (size_t) ( UINT_MAX - 2 );

    this->particlesBuffer->createParticleBuffer( gpuMemory );

}

template< typename T_DataVector, typename T_MethodsVector>
Particles<T_DataVector, T_MethodsVector>::~Particles( )
{
    delete this->particlesBuffer;
}

template< typename T_DataVector, typename T_MethodsVector>
void Particles<T_DataVector, T_MethodsVector>::synchronize( )
{
    this->particlesBuffer->deviceToHost( );
}

template< typename T_DataVector, typename T_MethodsVector>
void Particles<T_DataVector, T_MethodsVector>::syncToDevice( )
{
    this->particlesBuffer->hostToDevice( );
}

template< typename T_DataVector, typename T_MethodsVector>
void Particles<T_DataVector, T_MethodsVector>::init( FieldE &fieldE, FieldB &fieldB, FieldJ &fieldJ, int datasetID )
{
    this->fieldE = &fieldE;
    this->fieldB = &fieldB;
    this->fieldJurrent = &fieldJ;

    this->datasetID = datasetID;

    DataConnector::getInstance( ).registerData( *this, datasetID );
}

template< typename T_DataVector, typename T_MethodsVector>
void Particles<T_DataVector, T_MethodsVector>::update( uint32_t )
{
    typedef particlePusher::ParticlePusher ParticlePush;

    typedef typename GetMargin<fieldSolver::FieldToParticleInterpolation>::LowerMargin LowerMargin;
    typedef typename GetMargin<fieldSolver::FieldToParticleInterpolation>::UpperMargin UpperMargin;

    typedef PushParticlePerFrame<ParticlePush, MappingDesc::SuperCellSize,
        fieldSolver::NumericalCellType > FrameSolver;

    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        typename toTVec<LowerMargin>::type,
        typename toTVec<UpperMargin>::type
        > BlockArea;

    dim3 block( MappingDesc::SuperCellSize::getDataSpace( ) );

    __picKernelArea( kernelMoveAndMarkParticles<BlockArea>, this->cellDescription, CORE + BORDER )
        (block)
        ( this->getDeviceParticlesBox( ),
          this->fieldE->getDeviceDataBox( ),
          this->fieldB->getDeviceDataBox( ),
          FrameSolver( )
          );


    ParticlesBaseType::template shiftParticles < CORE + BORDER > ( );
    this->fillAllGaps( );

}

template< typename T_DataVector, typename T_MethodsVector>
void Particles<T_DataVector, T_MethodsVector>::reset( uint32_t )
{
    this->particlesBuffer->reset( );
}

template< typename T_DataVector, typename T_MethodsVector>
void Particles<T_DataVector, T_MethodsVector>::initFill( uint32_t currentStep )
{
    VirtualWindow window = MovingWindow::getInstance( ).getVirtualWindow( currentStep );
    PMACC_AUTO( simBox, SubGrid<simDim>::getInstance( ).getSimulationBox( ) );

    /*calculate real simulation area offset from the beginning of the simulation*/
    DataSpace<simDim> localCells = gridLayout.getDataSpaceWithoutGuarding( );
    DataSpace<simDim> gpuCellOffset = simBox.getGlobalOffset( );
    gpuCellOffset.y( ) += window.slides * localCells.y( );


    uint32_t seed = GridController<simDim>::getInstance( ).getGlobalSize( ) * FrameType::CommunicationTag
        + GridController<simDim>::getInstance( ).getGlobalRank( );
    seed ^= POSITION_SEED;
    dim3 block( MappingDesc::SuperCellSize::getDataSpace( ) );

    if ( gasProfile::GAS_ENABLED )
    {
        const DataSpace<simDim> globalNrOfCells = simBox.getGlobalSize( );

        __picKernelArea( kernelFillGridWithParticles, this->cellDescription, CORE + BORDER + GUARD )
            (block)
            ( this->particlesBuffer->getDeviceParticleBox( ),
              this->particlesBuffer->hasSendExchange( TOP ), gpuCellOffset, seed, globalNrOfCells.y( ) );
    }

    this->fillAllGaps( );

    log<picLog::SIMULATION_STATE > ( "Wait for init particles finished (y offset = %1%)" ) % gpuCellOffset.y( );
    __getTransactionEvent( ).waitForFinished( );
}

template< typename T_DataVector, typename T_MethodsVector>
template< typename t_DataVector, typename t_MethodsVector>
void Particles<T_DataVector, T_MethodsVector>::deviceCloneFrom( Particles<t_DataVector, t_MethodsVector> &src )
{
    dim3 block( TILE_SIZE );
    DataSpace<simDim> superCells = this->particlesBuffer->getSuperCellsCount( );

    __picKernelArea( kernelCloneParticles, this->cellDescription, CORE + BORDER + GUARD )
        (block) ( this->getDeviceParticlesBox( ), src.getDeviceParticlesBox( ) );
    log<picLog::SIMULATION_STATE > ( "start clone particles" );
    this->fillAllGaps( );

    log<picLog::SIMULATION_STATE > ( "Wait for clone particles finished" );
    __getTransactionEvent( ).waitForFinished( );
}

template< typename T_DataVector, typename T_MethodsVector>
void Particles<T_DataVector, T_MethodsVector>::deviceAddTemperature( float_X energy )
{
    dim3 block( MappingDesc::SuperCellSize::getDataSpace( ) );
    DataSpace<simDim> superCells = this->particlesBuffer->getSuperCellsCount( );

    uint32_t seed = GridController<simDim>::getInstance( ).getGlobalSize( ) * FrameType::CommunicationTag
        + GridController<simDim>::getInstance( ).getGlobalRank( );
    seed ^= TEMPERATURE_SEED;

    __picKernelArea( kernelAddTemperature, this->cellDescription, CORE + BORDER + GUARD )
        (block) ( this->getDeviceParticlesBox( ), energy, seed );

    log<picLog::SIMULATION_STATE > ( "Wait for addTemperature finished" );
    __getTransactionEvent( ).waitForFinished( );
}

template< typename T_DataVector, typename T_MethodsVector>
void Particles<T_DataVector, T_MethodsVector>::deviceSetDrift( uint32_t currentStep )
{
    VirtualWindow window = MovingWindow::getInstance( ).getVirtualWindow( currentStep );

    dim3 block( MappingDesc::SuperCellSize::getDataSpace( ) );

    PMACC_AUTO( simBox, SubGrid<simDim>::getInstance( ).getSimulationBox( ) );
    const DataSpace<simDim> localNrOfCells( simBox.getLocalSize( ) );
    const DataSpace<simDim> globalNrOfCells( simBox.getGlobalSize( ) );

    /* calculate real simulation area offset from the beginning of the simulation 
     */
    uint32_t simulationYCell = simBox.getGlobalOffset().y( ) +
        ( window.slides * localNrOfCells.y( ) );

    __picKernelArea( kernelSetDrift, this->cellDescription, CORE + BORDER + GUARD )
        (block)
        ( this->particlesBuffer->getDeviceParticleBox( ),
          simulationYCell,
          globalNrOfCells.y( ) );

    log<picLog::SIMULATION_STATE > ( "Wait for set drift finished" );
    __getTransactionEvent( ).waitForFinished( );
}

} // end namespace
