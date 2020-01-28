/* Copyright 2013-2020 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch, Felix Schmitt,
 *                     Alexander Grund, Sergei Bastrakov
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
#include "picongpu/particles/pusher/Traits.hpp"
#include "picongpu/particles/traits/GetExchangeMemCfg.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"

#include <pmacc/particles/memory/buffers/ParticlesBuffer.hpp>
#include "picongpu/particles/ParticlesInit.kernel"
#include <pmacc/mappings/simulation/GridController.hpp>

#include "picongpu/simulation/control/MovingWindow.hpp"

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
    picongpu::MappingDesc cellDescription,
    SimulationDataId datasetID
) :
    ParticlesBase<
        SpeciesParticleDescription,
        picongpu::MappingDesc,
        DeviceHeap
    >(
        heap,
        cellDescription
    ),
    m_datasetID( datasetID )
{
    using ExchangeMemCfg = GetExchangeMemCfg_t< Particles >;

    /* SPEC species exchange memory scaling
     *
     * The scaling factors are strongly coupled to the SPECBenchmark example
     * and a grid size of 128^3 cells per MPI rank.
     * By scaling required exchange memory size with the number of cells per MPI rank
     * we can avoid performance decreases due to double send of particles.
     */
    double numBaseCells = 128.0 * 128.0 * 128.0;
    double numGlobalCells = Environment<simDim>::get().SubGrid().getLocalDomain().size.productOfComponents();
    size_t memScalingFactorZ = static_cast< size_t >( std::ceil( numGlobalCells / numBaseCells / 2.0 ) );
    size_t memScalingFactorOther = static_cast< size_t >( std::ceil( numGlobalCells / numBaseCells / 4.0 ) );

    size_t sizeOfExchanges = 0u;

    const uint32_t commTag = pmacc::traits::GetUniqueTypeId<FrameType, uint32_t>::uid() + SPECIES_FIRSTTAG;
    log<picLog::MEMORY > ( "communication tag for species %1%: %2%" ) % FrameType::getName( ) % commTag;

    this->particlesBuffer->addExchange( Mask( LEFT ) + Mask( RIGHT ),
                                        ExchangeMemCfg::BYTES_EXCHANGE_X * memScalingFactorOther,
                                        commTag);
    sizeOfExchanges += ExchangeMemCfg::BYTES_EXCHANGE_X * 2u * memScalingFactorOther;

    this->particlesBuffer->addExchange( Mask( TOP ) + Mask( BOTTOM ),
                                        ExchangeMemCfg::BYTES_EXCHANGE_Y * memScalingFactorOther,
                                        commTag);
    sizeOfExchanges += ExchangeMemCfg::BYTES_EXCHANGE_Y * 2u * memScalingFactorOther;

    //edges of the simulation area
    this->particlesBuffer->addExchange( Mask( RIGHT + TOP ) + Mask( LEFT + TOP ) +
                                        Mask( LEFT + BOTTOM ) + Mask( RIGHT + BOTTOM ), ExchangeMemCfg::BYTES_EDGES * memScalingFactorOther,
                                        commTag);
    sizeOfExchanges += ExchangeMemCfg::BYTES_EDGES * 4u * memScalingFactorOther;

#if(SIMDIM==DIM3)
    this->particlesBuffer->addExchange( Mask( FRONT ) + Mask( BACK ), ExchangeMemCfg::BYTES_EXCHANGE_Z * memScalingFactorZ,
                                        commTag);
    sizeOfExchanges += ExchangeMemCfg::BYTES_EXCHANGE_Z * 2u * memScalingFactorZ;

    //edges of the simulation area
    this->particlesBuffer->addExchange( Mask( FRONT + TOP ) + Mask( BACK + TOP ) +
                                        Mask( FRONT + BOTTOM ) + Mask( BACK + BOTTOM ),
                                        ExchangeMemCfg::BYTES_EDGES * memScalingFactorOther,
                                        commTag);
    sizeOfExchanges += ExchangeMemCfg::BYTES_EDGES * 4u * memScalingFactorOther;

    this->particlesBuffer->addExchange( Mask( FRONT + RIGHT ) + Mask( BACK + RIGHT ) +
                                        Mask( FRONT + LEFT ) + Mask( BACK + LEFT ),
                                        ExchangeMemCfg::BYTES_EDGES * memScalingFactorOther,
                                        commTag);
    sizeOfExchanges += ExchangeMemCfg::BYTES_EDGES * 4u * memScalingFactorOther;

    //corner of the simulation area
    this->particlesBuffer->addExchange( Mask( TOP + FRONT + RIGHT ) + Mask( TOP + BACK + RIGHT ) +
                                        Mask( BOTTOM + FRONT + RIGHT ) + Mask( BOTTOM + BACK + RIGHT ),
                                        ExchangeMemCfg::BYTES_CORNER * memScalingFactorOther,
                                        commTag);
    sizeOfExchanges += ExchangeMemCfg::BYTES_CORNER * 4u * memScalingFactorOther;

    this->particlesBuffer->addExchange( Mask( TOP + FRONT + LEFT ) + Mask( TOP + BACK + LEFT ) +
                                        Mask( BOTTOM + FRONT + LEFT ) + Mask( BOTTOM + BACK + LEFT ),
                                        ExchangeMemCfg::BYTES_CORNER,
                                        commTag);
    sizeOfExchanges += ExchangeMemCfg::BYTES_CORNER * 4u * memScalingFactorOther;
#endif

    /* The buffer size must be multiplied by two because PMacc generates a send
     * and receive buffer for each direction.
     */
    sizeOfExchanges *= 2u;

    constexpr size_t byteToMiB = 1024u * 1024u;

    log< picLog::MEMORY >( "size for all exchange of species %1% = %2% MiB" ) %
        FrameType::getName( ) %
        ( static_cast< float_64 >( sizeOfExchanges ) / static_cast< float_64 >( byteToMiB ) );
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

/** Launcher of the particle push
 *
 * @tparam T_Pusher pusher type
 * @tparam T_isComposite if the pusher is composite
 */
template<
    typename T_Pusher,
    bool T_isComposite = particles::pusher::IsComposite< T_Pusher >::value
>
struct PushLauncher;

/** Launcher of the particle push for non-composite pushers
 *
 * @tparam T_Pusher pusher type
 */
template< typename T_Pusher >
struct PushLauncher<
    T_Pusher,
    false
>
{
    /** Launch the pusher for all particles of a species
     *
     * @tparam T_Particles particles type
     * @param currentStep current time iteration
     */
    template< typename T_Particles >
    void operator()(
        T_Particles && particles,
        uint32_t const currentStep
    ) const
    {
        particles.template push< T_Pusher >( currentStep );
    }
};

/** Launcher of the particle push for composite pushers
 *
 * @tparam T_Pusher pusher type
 */
template< typename T_CompositePusher >
struct PushLauncher<
    T_CompositePusher,
    true
>
{
    /** Launch the pusher for all particles of a species
     *
     * @tparam T_Particles particles type
     * @param currentStep current time iteration
     */
    template< typename T_Particles >
    void operator()(
        T_Particles && particles,
        uint32_t const currentStep
    ) const
    {
        /* Here we check for the active pusher and only call PushLauncher for
         * that one. Note that we still instantiate both templates, but this
         * should be fine as both pushers are eventually getting used (otherwise
         * using the composite does not make sense).
         */
        auto activePusherIdx = T_CompositePusher::activePusherIdx( currentStep );
        if( activePusherIdx == 1 )
            PushLauncher< typename T_CompositePusher::FirstPusher >{}(
                particles,
                currentStep
            );
        else if( activePusherIdx == 2 )
            PushLauncher< typename T_CompositePusher::SecondPusher >{}(
                particles,
                currentStep
            );
    }
};

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
>::update( uint32_t const currentStep )
{
    using PusherAlias = typename GetFlagType<FrameType,particlePusher<> >::type;
    using ParticlePush = typename pmacc::traits::Resolve<PusherAlias>::type;
    // Because of composite pushers, we have to defer using the launcher
    PushLauncher< ParticlePush >{}(
        *this,
        currentStep
    );
}

/** Do the particle push stage using the given pusher
 *
 * @tparam T_Pusher non-composite pusher type
 * @param currentStep current time iteration
 */
template<
    typename T_Name,
    typename T_Flags,
    typename T_Attributes
>
template< typename T_Pusher >
void
Particles<
    T_Name,
    T_Flags,
    T_Attributes
>::push( uint32_t const currentStep )
{
    PMACC_CASSERT_MSG(
        _internal_error_particle_push_instantiated_for_composite_pusher,
        particles::pusher::IsComposite< T_Pusher >::type::value == false
    );

    using InterpolationScheme = typename pmacc::traits::Resolve<
        typename GetFlagType<
            FrameType,
            interpolation< >
        >::type
    >::type;

    using FrameSolver = PushParticlePerFrame<
        T_Pusher,
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

    /* Adjust interpolation area in particle pusher to allow sub-stepping pushes.
     * Here were provide an actual pusher and use its actual margins
     */
    using LowerMargin = typename GetLowerMarginForPusher<
        Particles,
        T_Pusher
    >::type;
    using UpperMargin = typename GetUpperMarginForPusher<
        Particles,
        T_Pusher
    >::type;

    using BlockArea = SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        LowerMargin,
        UpperMargin
    >;

    AreaMapping<
        CORE + BORDER,
        picongpu::MappingDesc
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
        currentStep,
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
        picongpu::MappingDesc
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
    typename T_ManipulateFunctor,
    typename T_SrcFilterFunctor
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
    T_ManipulateFunctor& manipulatorFunctor,
    T_SrcFilterFunctor& srcFilterFunctor
)
{
    log< picLog::SIMULATION_STATE > ( "clone species %1%" ) % FrameType::getName( );

    AreaMapping<CORE + BORDER, picongpu::MappingDesc> mapper(this->cellDescription);

    constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
           pmacc::math::CT::volume< SuperCellSize >::type::value
    >::value;

    PMACC_KERNEL( KernelDeriveParticles< numWorkers >{ } )(
        mapper.getGridDim(),
        numWorkers
    )(
        this->getDeviceParticlesBox( ),
        src.getDeviceParticlesBox( ),
        manipulatorFunctor,
        srcFilterFunctor,
        mapper
    );
    this->fillAllGaps( );
}

} // namespace picongpu
