/**
 * Copyright 2013-2017 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch, Felix Schmitt,
 *                     Alexander Grund
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

#include "simulation_defines.hpp"
#include "Particles.hpp"

#include "particles/Particles.kernel"

#include "dataManagement/DataConnector.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"

#include "particles/memory/buffers/ParticlesBuffer.hpp"
#include "ParticlesInit.kernel"
#include "mappings/simulation/GridController.hpp"

#include "simulationControl/MovingWindow.hpp"

#include "fields/numericalCellTypes/YeeCell.hpp"

#include "traits/GetUniqueTypeId.hpp"
#include "traits/Resolve.hpp"
#include "particles/traits/GetMarginPusher.hpp"

#include <iostream>
#include <limits>

namespace picongpu
{


using namespace PMacc;

template<
    typename T_Name,
    typename T_Attributes,
    typename T_Flags
>
Particles<
    T_Name,
    T_Attributes,
    T_Flags
>::Particles(
    GridLayout<simDim>,
    MappingDesc cellDescription,
    SimulationDataId datasetID
) :
    ParticlesBase<
        SpeciesParticleDescription,
        MappingDesc
    >( cellDescription ),
    fieldB( NULL ),
    fieldE( NULL ),
    m_datasetID( datasetID )
{
    size_t sizeOfExchanges = 2 * 2 * ( BYTES_EXCHANGE_X + BYTES_EXCHANGE_Y + BYTES_EXCHANGE_Z ) + BYTES_EXCHANGE_X * 2 * 8;

    log<picLog::MEMORY > ( "size for all exchange = %1% MiB" ) % ( (float_64) sizeOfExchanges / 1024. / 1024. );

    const uint32_t commTag = PMacc::traits::GetUniqueTypeId<FrameType, uint32_t>::uid() + SPECIES_FIRSTTAG;
    log<picLog::MEMORY > ( "communication tag for species %1%: %2%" ) % FrameType::getName( ) % commTag;

    this->particlesBuffer->addExchange( Mask( LEFT ) + Mask( RIGHT ),
                                        BYTES_EXCHANGE_X,
                                        commTag);
    this->particlesBuffer->addExchange( Mask( TOP ) + Mask( BOTTOM ),
                                        BYTES_EXCHANGE_Y,
                                        commTag);
    //edges of the simulation area
    this->particlesBuffer->addExchange( Mask( RIGHT + TOP ) + Mask( LEFT + TOP ) +
                                        Mask( LEFT + BOTTOM ) + Mask( RIGHT + BOTTOM ), BYTES_EDGES,
                                        commTag);

#if(SIMDIM==DIM3)
    this->particlesBuffer->addExchange( Mask( FRONT ) + Mask( BACK ), BYTES_EXCHANGE_Z,
                                        commTag);
    //edges of the simulation area
    this->particlesBuffer->addExchange( Mask( FRONT + TOP ) + Mask( BACK + TOP ) +
                                        Mask( FRONT + BOTTOM ) + Mask( BACK + BOTTOM ),
                                        BYTES_EDGES,
                                        commTag);
    this->particlesBuffer->addExchange( Mask( FRONT + RIGHT ) + Mask( BACK + RIGHT ) +
                                        Mask( FRONT + LEFT ) + Mask( BACK + LEFT ),
                                        BYTES_EDGES,
                                        commTag);
    //corner of the simulation area
    this->particlesBuffer->addExchange( Mask( TOP + FRONT + RIGHT ) + Mask( TOP + BACK + RIGHT ) +
                                        Mask( BOTTOM + FRONT + RIGHT ) + Mask( BOTTOM + BACK + RIGHT ),
                                        BYTES_CORNER,
                                        commTag);
    this->particlesBuffer->addExchange( Mask( TOP + FRONT + LEFT ) + Mask( TOP + BACK + LEFT ) +
                                        Mask( BOTTOM + FRONT + LEFT ) + Mask( BOTTOM + BACK + LEFT ),
                                        BYTES_CORNER,
                                        commTag);
#endif
}

template<
    typename T_Name,
    typename T_Attributes,
    typename T_Flags
>
void
Particles<
    T_Name,
    T_Attributes,
    T_Flags
>::createParticleBuffer( )
{
    this->particlesBuffer->createParticleBuffer( );
}

template<
    typename T_Name,
    typename T_Attributes,
    typename T_Flags
>
SimulationDataId
Particles<
    T_Name,
    T_Attributes,
    T_Flags
>::getUniqueId( )
{
    return m_datasetID;
}

template<
    typename T_Name,
    typename T_Attributes,
    typename T_Flags
>
void
Particles<
    T_Name,
    T_Attributes,
    T_Flags
>::synchronize( )
{
    this->particlesBuffer->deviceToHost();
}

template<
    typename T_Name,
    typename T_Attributes,
    typename T_Flags
>
void
Particles<
    T_Name,
    T_Attributes,
    T_Flags
>::syncToDevice( )
{

}

template<
    typename T_Name,
    typename T_Attributes,
    typename T_Flags
>
void
Particles<
    T_Name,
    T_Attributes,
    T_Flags
>::init( FieldE &fieldE, FieldB &fieldB )
{
    this->fieldE = &fieldE;
    this->fieldB = &fieldB;

    Environment<>::get( ).DataConnector( ).registerData( *this );
}

template<
    typename T_Name,
    typename T_Attributes,
    typename T_Flags
>
void
Particles<
    T_Name,
    T_Attributes,
    T_Flags
>::update(uint32_t )
{
    typedef typename GetFlagType<FrameType,particlePusher<> >::type PusherAlias;
    typedef typename PMacc::traits::Resolve<PusherAlias>::type ParticlePush;

    typedef typename PMacc::traits::Resolve<
        typename GetFlagType<FrameType,interpolation<> >::type
        >::type InterpolationScheme;

    typedef PushParticlePerFrame<ParticlePush, MappingDesc::SuperCellSize,
        InterpolationScheme > FrameSolver;

    // adjust interpolation area in particle pusher to allow sub-sampling pushes
    typedef typename GetLowerMarginPusher<Particles>::type LowerMargin;
    typedef typename GetUpperMarginPusher<Particles>::type UpperMargin;

    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        LowerMargin,
        UpperMargin
        > BlockArea;

    auto block = MappingDesc::SuperCellSize::toRT();

    AreaMapping<CORE+BORDER,MappingDesc> mapper(this->cellDescription);
    PMACC_KERNEL( KernelMoveAndMarkParticles<BlockArea>{} )
        (mapper.getGridDim(), block)
        ( this->getDeviceParticlesBox( ),
          this->fieldE->getDeviceDataBox( ),
          this->fieldB->getDeviceDataBox( ),
          FrameSolver( ),
          mapper
          );

    ParticlesBaseType::template shiftParticles < CORE + BORDER > ( );
}

template<
    typename T_Name,
    typename T_Attributes,
    typename T_Flags
>
template<typename T_DensityFunctor, typename T_PositionFunctor>
void
Particles<
    T_Name,
    T_Attributes,
    T_Flags
>::initDensityProfile(
    T_DensityFunctor& densityFunctor,
    T_PositionFunctor& positionFunctor,
    const uint32_t currentStep
)
{
    log<picLog::SIMULATION_STATE > ( "initialize density profile for species %1%" ) % FrameType::getName( );

    const uint32_t numSlides = MovingWindow::getInstance( ).getSlideCounter( currentStep );
    const SubGrid<simDim>& subGrid = Environment<simDim>::get( ).SubGrid( );
    DataSpace<simDim> localCells = subGrid.getLocalDomain( ).size;
    DataSpace<simDim> totalGpuCellOffset = subGrid.getLocalDomain( ).offset;
    totalGpuCellOffset.y( ) += numSlides * localCells.y( );

    auto block = MappingDesc::SuperCellSize::toRT( );
    AreaMapping<CORE+BORDER,MappingDesc> mapper(this->cellDescription);
    PMACC_KERNEL( KernelFillGridWithParticles< Particles >{} )
        (mapper.getGridDim(), block)
        ( densityFunctor, positionFunctor, totalGpuCellOffset, this->particlesBuffer->getDeviceParticleBox( ), mapper );


    this->fillAllGaps( );
}

template<
    typename T_Name,
    typename T_Attributes,
    typename T_Flags
>
template<
    typename T_SrcName,
    typename T_SrcAttributes,
    typename T_SrcFlags,
    typename T_ManipulateFunctor
>
void
Particles<
    T_Name,
    T_Attributes,
    T_Flags
>::deviceDeriveFrom(
    Particles<
        T_SrcName,
        T_SrcAttributes,
        T_SrcFlags
    >& src,
    T_ManipulateFunctor& functor
)
{
    auto block = PMacc::math::CT::volume<SuperCellSize>::type::value;

    log<picLog::SIMULATION_STATE > ( "clone species %1%" ) % FrameType::getName( );
    AreaMapping<CORE + BORDER, MappingDesc> mapper(this->cellDescription);
    PMACC_KERNEL( KernelDeriveParticles{} )
        (mapper.getGridDim(), block) ( this->getDeviceParticlesBox( ), src.getDeviceParticlesBox( ), functor, mapper );
    this->fillAllGaps( );
}

template<
    typename T_Name,
    typename T_Attributes,
    typename T_Flags
>
template< typename T_Functor>
void
Particles<
    T_Name,
    T_Attributes,
    T_Flags
>::manipulateAllParticles( uint32_t currentStep, T_Functor& functor )
{

    auto block = MappingDesc::SuperCellSize::toRT( );

    AreaMapping<CORE + BORDER, MappingDesc> mapper(this->cellDescription);
    PMACC_KERNEL( KernelManipulateAllParticles{} )
        (mapper.getGridDim(), block)
        ( this->particlesBuffer->getDeviceParticleBox( ),
          functor,
          mapper
        );
}

} // end namespace
