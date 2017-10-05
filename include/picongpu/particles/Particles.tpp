/* Copyright 2013-2017 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch, Felix Schmitt,
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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/particles/Particles.hpp"

#include "picongpu/particles/Particles.kernel"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"

#include <pmacc/particles/memory/buffers/ParticlesBuffer.hpp>
#include "picongpu/particles/ParticlesInit.kernel"
#include <pmacc/mappings/simulation/GridController.hpp>

#include "picongpu/simulationControl/MovingWindow.hpp"

#include "picongpu/fields/numericalCellTypes/YeeCell.hpp"
#include "picongpu/particles/traits/GetMarginPusher.hpp"

#include <pmacc/traits/GetUniqueTypeId.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>

#include <iostream>
#include <limits>
#include <memory>


namespace picongpu
{


using namespace pmacc;

template<
    typename T_Name,
    typename T_Flags,
    typename T_Attributes
>
Particles<
    T_Name,
    T_Flags,
    T_Attributes
>::Particles(
    const std::shared_ptr<DeviceHeap>& heap,
    MappingDesc cellDescription,
    SimulationDataId datasetID
) :
    ParticlesBase<
        SpeciesParticleDescription,
        MappingDesc,
        DeviceHeap
    >(
        heap,
        cellDescription
    ),
    m_datasetID( datasetID )
{
    size_t sizeOfExchanges = 2 * 2 * ( BYTES_EXCHANGE_X + BYTES_EXCHANGE_Y + BYTES_EXCHANGE_Z ) + BYTES_EXCHANGE_X * 2 * 8;

    log<picLog::MEMORY > ( "size for all exchange = %1% MiB" ) % ( (float_64) sizeOfExchanges / 1024. / 1024. );

    const uint32_t commTag = pmacc::traits::GetUniqueTypeId<FrameType, uint32_t>::uid() + SPECIES_FIRSTTAG;
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
    typename T_Flags,
    typename T_Attributes
>
void
Particles<
    T_Name,
    T_Flags,
    T_Attributes
>::createParticleBuffer( )
{
    this->particlesBuffer->createParticleBuffer( );
}

template<
    typename T_Name,
    typename T_Flags,
    typename T_Attributes
>
SimulationDataId
Particles<
    T_Name,
    T_Flags,
    T_Attributes
>::getUniqueId( )
{
    return m_datasetID;
}

template<
    typename T_Name,
    typename T_Flags,
    typename T_Attributes
>
void
Particles<
    T_Name,
    T_Flags,
    T_Attributes
>::synchronize( )
{
    this->particlesBuffer->deviceToHost();
}

template<
    typename T_Name,
    typename T_Flags,
    typename T_Attributes
>
void
Particles<
    T_Name,
    T_Flags,
    T_Attributes
>::syncToDevice( )
{

}

template<
    typename T_Name,
    typename T_Flags,
    typename T_Attributes
>
void
Particles<
    T_Name,
    T_Flags,
    T_Attributes
>::update(uint32_t )
{
    using PusherAlias = typename GetFlagType<FrameType,particlePusher<> >::type;
    using ParticlePush = typename pmacc::traits::Resolve<PusherAlias>::type;

    using InterpolationScheme = typename pmacc::traits::Resolve<
        typename GetFlagType<
            FrameType,
            interpolation< >
        >::type
    >::type;

    using FrameSolver = PushParticlePerFrame<
        ParticlePush,
        MappingDesc::SuperCellSize,
        InterpolationScheme
    >;

    DataConnector & dc = Environment< >::get( ).DataConnector( );
    auto fieldE = dc.get< FieldE >(
        FieldE::getName(),
        true
    );
    auto fieldB = dc.get< FieldB >(
        FieldB::getName(),
        true
    );

    // adjust interpolation area in particle pusher to allow sub-sampling pushes
    using LowerMargin = typename GetLowerMarginPusher< Particles >::type;
    using UpperMargin = typename GetUpperMarginPusher< Particles >::type;

    using BlockArea = SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        LowerMargin,
        UpperMargin
    >;

    AreaMapping<
        CORE + BORDER,
        MappingDesc
    > mapper( this->cellDescription );

    constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
        pmacc::math::CT::volume< SuperCellSize >::type::value
    >::value;

    PMACC_KERNEL( KernelMoveAndMarkParticles< numWorkers, BlockArea >{ } )(
        mapper.getGridDim(),
        numWorkers
    )(
        this->getDeviceParticlesBox( ),
        fieldE->getDeviceDataBox( ),
        fieldB->getDeviceDataBox( ),
        FrameSolver( ),
        mapper
    );

    dc.releaseData( FieldE::getName() );
    dc.releaseData( FieldB::getName() );

    ParticlesBaseType::template shiftParticles < CORE + BORDER > ( );
}

template<
    typename T_Name,
    typename T_Flags,
    typename T_Attributes
>
template<
    typename T_DensityFunctor,
    typename T_PositionFunctor
>
void
Particles<
    T_Name,
    T_Flags,
    T_Attributes
>::initDensityProfile(
    T_DensityFunctor& densityFunctor,
    T_PositionFunctor& positionFunctor,
    const uint32_t currentStep
)
{
    log<picLog::SIMULATION_STATE >( "initialize density profile for species %1%" ) % FrameType::getName( );

    uint32_t const numSlides = MovingWindow::getInstance( ).getSlideCounter( currentStep );
    SubGrid< simDim > const & subGrid = Environment< simDim >::get( ).SubGrid( );
    DataSpace< simDim > localCells = subGrid.getLocalDomain( ).size;
    DataSpace< simDim > totalGpuCellOffset = subGrid.getLocalDomain( ).offset;
    totalGpuCellOffset.y( ) += numSlides * localCells.y( );

    constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
        pmacc::math::CT::volume< SuperCellSize >::type::value
    >::value;

    AreaMapping<
        CORE + BORDER,
        MappingDesc
    > mapper( this->cellDescription );
    PMACC_KERNEL(
        KernelFillGridWithParticles<
            numWorkers,
            Particles
        >{}
    )
    (
        mapper.getGridDim( ),
        numWorkers
    )
    (
        densityFunctor,
        positionFunctor,
        totalGpuCellOffset,
        this->particlesBuffer->getDeviceParticleBox( ),
        mapper
    );

    this->fillAllGaps( );
}

template<
    typename T_Name,
    typename T_Flags,
    typename T_Attributes
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
    T_Flags,
    T_Attributes
>::deviceDeriveFrom(
    Particles<
        T_SrcName,
        T_SrcAttributes,
        T_SrcFlags
    >& src,
    T_ManipulateFunctor& functor
)
{
    log< picLog::SIMULATION_STATE > ( "clone species %1%" ) % FrameType::getName( );

    AreaMapping<CORE + BORDER, MappingDesc> mapper(this->cellDescription);

    constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
           pmacc::math::CT::volume< SuperCellSize >::type::value
    >::value;

    PMACC_KERNEL( KernelDeriveParticles< numWorkers >{ } )(
        mapper.getGridDim(),
        numWorkers
    )(
        this->getDeviceParticlesBox( ),
        src.getDeviceParticlesBox( ),
        functor,
        mapper
    );
    this->fillAllGaps( );
}

template<
    typename T_Name,
    typename T_Flags,
    typename T_Attributes
>
template< typename T_Functor >
void
Particles<
    T_Name,
    T_Flags,
    T_Attributes
>::manipulateAllParticles(
    uint32_t currentStep,
    T_Functor & functor
)
{
    AreaMapping<
        CORE + BORDER,
        MappingDesc
    > mapper( this->cellDescription );

    constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
        pmacc::math::CT::volume< SuperCellSize >::type::value
    >::value;

    PMACC_KERNEL( KernelManipulateAllParticles< numWorkers >{ } )(
        mapper.getGridDim( ),
        numWorkers
    )(
        this->particlesBuffer->getDeviceParticleBox( ),
        functor,
        mapper
    );
}

} // namespace picongpu
