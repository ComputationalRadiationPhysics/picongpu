/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch, Felix Schmitt,
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

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/particles/Particles.hpp"
#include "picongpu/particles/Particles.kernel"
#include "picongpu/particles/ParticlesInit.kernel"
#include "picongpu/particles/boundary/Apply.hpp"
#include "picongpu/particles/pusher/Traits.hpp"
#include "picongpu/particles/traits/GetExchangeMemCfg.hpp"
#include "picongpu/particles/traits/GetMarginPusher.hpp"
#include "picongpu/simulation/control/MovingWindow.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/particles/memory/buffers/ParticlesBuffer.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace picongpu
{
    using namespace pmacc;

    namespace detail
    {
        /** Calculate the scaling factor for each direction.
         *
         * The scaling factor is derived from the reference size of the local domain and a scaling factor provided by
         * the user.
         *
         * @tparam T_ExchangeMemCfg exchange configuration for a species
         * @tparam T_Sfinae Type for conditionally specialization (no input parameter)
         * @{
         */
        template<typename T_ExchangeMemCfg, typename T_Sfinae = void>
        struct DirScalingFactor
        {
            //! @return orthogonal contribution factor to scale the amount of memory for each direction
            static floatD_64 getOrtho()
            {
                return floatD_64::create(1.0);
            }

            //! @return parallel contribution factor to scale the amount of memory for each direction
            //! (userScalingFactor)
            static floatD_64 getPara()
            {
                return floatD_64::create(1.0);
            }
        };

        /** Specialization for species with exchange memory information which provides
         * DIR_SCALING_FACTOR and REF_LOCAL_DOM_SIZE
         */
        template<typename T_ExchangeMemCfg>
        struct DirScalingFactor<
            T_ExchangeMemCfg,
            std::void_t<
                decltype(std::declval<T_ExchangeMemCfg>().DIR_SCALING_FACTOR),
                typename T_ExchangeMemCfg::REF_LOCAL_DOM_SIZE>>
        {
            static floatD_64 getOrtho()
            {
                auto baseLocalCells = T_ExchangeMemCfg::REF_LOCAL_DOM_SIZE::toRT();
                auto localDomSize = Environment<simDim>::get().SubGrid().getLocalDomain().size;

                auto scale = floatD_64::create(1.0);
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    // set to local domain size in case there is no base volume defined
                    if(baseLocalCells[d] <= 0)
                        baseLocalCells[d] = localDomSize[d];

                    scale[d] = std::max(float_64(localDomSize[d]) / float_64(baseLocalCells[d]), 1.0);
                }

                return scale;
            }

            static floatD_64 getPara()
            {
                auto userScalingFactor = T_ExchangeMemCfg{}.DIR_SCALING_FACTOR;
                floatD_64 scale;
                for(uint32_t d = 0; d < simDim; ++d)
                    scale[d] = userScalingFactor[d];

                return scale;
            }
        };

        //! @}
    } // namespace detail

    template<typename T_Name, typename T_Flags, typename T_Attributes>
    size_t Particles<T_Name, T_Flags, T_Attributes>::exchangeMemorySize(uint32_t ex) const
    {
        // no communication direction
        if(ex == 0u)
            return 0u;

        using ExchangeMemCfg = GetExchangeMemCfg_t<Particles>;
        // scaling factors, base and local cell sizes for each direction
        auto orthoScalingFactor = ::picongpu::detail::DirScalingFactor<ExchangeMemCfg>::getOrtho();
        auto paraScalingFactor = ::picongpu::detail::DirScalingFactor<ExchangeMemCfg>::getPara();

        /* type of the exchange direction
         * 1 = plane
         * 2 = edge
         * 3 = corner
         */
        uint32_t relDirType = 0u;

        // scaling factor for the current exchange
        float_64 exchangeScalingFactor = 1.0;

        auto relDir = Mask::getRelativeDirections<simDim>(ex);
        for(uint32_t d = 0; d < simDim; ++d)
        {
            // calculate the exchange type
            relDirType += std::abs(relDir[d]);
            if(relDir[d] == 0) // scale up by factors orthorgonal to exchange dir
                exchangeScalingFactor *= orthoScalingFactor[d];
            else // apply user scaling in exchange dir
                exchangeScalingFactor *= paraScalingFactor[d];
        }
        exchangeScalingFactor = std::max(exchangeScalingFactor, 1.0);

        size_t exchangeBytes = 0;

        using ExchangeMemCfg = GetExchangeMemCfg_t<Particles>;

        // it is a exachange
        if(relDirType == 1u)
        {
            // x, y, z, edge, corner
            pmacc::math::Vector<uint32_t, 3u> requiredMem(
                ExchangeMemCfg::BYTES_EXCHANGE_X,
                ExchangeMemCfg::BYTES_EXCHANGE_Y,
                ExchangeMemCfg::BYTES_EXCHANGE_Z);

            for(uint32_t d = 0; d < simDim; ++d)
                if(std::abs(relDir[d]) == 1)
                {
                    exchangeBytes = requiredMem[d];
                    break;
                }
        }
        // it is an edge
        else if(relDirType == 2u)
            exchangeBytes = ExchangeMemCfg::BYTES_EDGES;
        // it is a corner
        else
            exchangeBytes = ExchangeMemCfg::BYTES_CORNER;

        /* PIConGPU default case is running with 32bit (4byte) precision. In case double precision is used the particle
         * buffers must be twice as large to avoid that a communication must be split into multiple sends.
         */
        double precisionMultiplier = static_cast<double>(sizeof(float_X)) / static_cast<double>(sizeof(float));
        // using double to calculate the memory size is fine, double can precise store integer values up too 2^53
        return exchangeBytes * exchangeScalingFactor * precisionMultiplier;
    }

    template<typename T_Name, typename T_Flags, typename T_Attributes>
    Particles<T_Name, T_Flags, T_Attributes>::Particles(
        const std::shared_ptr<DeviceHeap>& heap,
        picongpu::MappingDesc cellDescription,
        SimulationDataId datasetID)
        : ParticlesBase<SpeciesParticleDescription, picongpu::MappingDesc, DeviceHeap>(heap, cellDescription)
        , m_datasetID(datasetID)
    {
        constexpr bool particleHasShape = pmacc::traits::HasIdentifier<FrameType, shape<>>::type::value;
        if constexpr(particleHasShape)
        {
            constexpr auto particleAssignmentShapeSupport = GetShape<Particles>::type::ChargeAssignment::support;
            static_assert(
                particleAssignmentShapeSupport > 0,
                "A particle shape must have a support larger than zero. Please use a higher order shape. If you "
                "need a pointwise particle use NGP shape.");
        }

        size_t sizeOfExchanges = 0u;

        const uint32_t commTag = pmacc::traits::getUniqueId();
        log<picLog::MEMORY>("communication tag for species %1%: %2%") % FrameType::getName() % commTag;

        auto const numExchanges = NumberOfExchanges<simDim>::value;
        for(uint32_t exchange = 1u; exchange < numExchanges; ++exchange)
        {
            auto mask = Mask(exchange);
            auto mem = exchangeMemorySize(exchange);

            this->particlesBuffer->addExchange(mask, mem, commTag);
            /* The buffer size must be multiplied by two because PMacc generates a send
             * and receive buffer for each direction.
             */
            sizeOfExchanges += mem * 2u;
        };

        constexpr size_t byteToMiB = 1024u * 1024u;

        log<picLog::MEMORY>("size for all exchange of species %1% = %2% MiB") % FrameType::getName()
            % (static_cast<float_64>(sizeOfExchanges) / static_cast<float_64>(byteToMiB));
    }

    template<typename T_Name, typename T_Flags, typename T_Attributes>
    void Particles<T_Name, T_Flags, T_Attributes>::createParticleBuffer()
    {
        this->particlesBuffer->createParticleBuffer();
    }

    template<typename T_Name, typename T_Flags, typename T_Attributes>
    SimulationDataId Particles<T_Name, T_Flags, T_Attributes>::getUniqueId()
    {
        return m_datasetID;
    }

    template<typename T_Name, typename T_Flags, typename T_Attributes>
    void Particles<T_Name, T_Flags, T_Attributes>::synchronize()
    {
        this->particlesBuffer->deviceToHost();
    }

    template<typename T_Name, typename T_Flags, typename T_Attributes>
    void Particles<T_Name, T_Flags, T_Attributes>::syncToDevice()
    {
    }

    /** Launcher of the particle push
     *
     * @tparam T_Pusher pusher type
     * @tparam T_isComposite if the pusher is composite
     */
    template<typename T_Pusher>
    struct PushLauncher
    {
        /** Launch the pusher for all particles of a species
         *
         * @tparam T_Particles particles type
         * @param currentStep current time iteration
         */
        template<typename T_Particles>
        void operator()(T_Particles&& particles, uint32_t const currentStep) const
        {
            constexpr bool isCompositePusher = particles::pusher::IsComposite<T_Pusher>::value;
            if constexpr(isCompositePusher)
            {
                /* Here we check for the active pusher and only call PushLauncher for
                 * that one. Note that we still instantiate both templates, but this
                 * should be fine as both pushers are eventually getting used (otherwise
                 * using the composite does not make sense).
                 */
                auto activePusherIdx = T_Pusher::activePusherIdx(currentStep);
                if(activePusherIdx == 1)
                    PushLauncher<typename T_Pusher::FirstPusher>{}(particles, currentStep);
                else if(activePusherIdx == 2)
                    PushLauncher<typename T_Pusher::SecondPusher>{}(particles, currentStep);
            }
            else
                particles.template push<T_Pusher>(currentStep);
        }
    };

    template<typename T_Name, typename T_Flags, typename T_Attributes>
    void Particles<T_Name, T_Flags, T_Attributes>::update(uint32_t const currentStep)
    {
        using PusherAlias = typename pmacc::traits::GetFlagType<FrameType, particlePusher<>>::type;
        using ParticlePush = typename pmacc::traits::Resolve<PusherAlias>::type;
        // Because of composite pushers, we have to defer using the launcher
        PushLauncher<ParticlePush>{}(*this, currentStep);
    }

    template<typename T_Name, typename T_Flags, typename T_Attributes>
    void Particles<T_Name, T_Flags, T_Attributes>::applyBoundary(uint32_t const currentStep)
    {
        using HasMomentum = typename pmacc::traits::HasIdentifier<FrameType, momentum>::type;
        /* We have to templatize lambda parameter to defer its instantiation.
         * Otherwise it would have been instantiated for all species, not just supported ones.
         */
        if constexpr(HasMomentum::value)
            particles::boundary::apply(*this, currentStep);
    }

    /** Do the particle push stage using the given pusher
     *
     * @tparam T_Pusher non-composite pusher type
     * @param currentStep current time iteration
     */
    template<typename T_Name, typename T_Flags, typename T_Attributes>
    template<typename T_Pusher>
    void Particles<T_Name, T_Flags, T_Attributes>::push(uint32_t const currentStep)
    {
        /* Particle push logic requires that a particle cannot pass more than a cell in a time step.
         * For 2d this concerns only steps in x, y.
         */
        constexpr auto dz = (simDim == 3) ? CELL_DEPTH : std::numeric_limits<float_X>::infinity();
        constexpr auto minCellSize = std::min({CELL_WIDTH, CELL_HEIGHT, dz});
        PMACC_CASSERT_MSG(
            Particle_in_pusher_cannot_pass_more_than_1_cell_per_time_step____check_your_grid_param_file,
            (SPEED_OF_LIGHT * DELTA_T / minCellSize <= 1.0) && sizeof(T_Pusher*) != 0);

        PMACC_CASSERT_MSG(
            _internal_error_particle_push_instantiated_for_composite_pusher,
            particles::pusher::IsComposite<T_Pusher>::type::value == false);

        using InterpolationScheme = typename pmacc::traits::Resolve<
            typename pmacc::traits::GetFlagType<FrameType, interpolation<>>::type>::type;

        using FrameSolver = PushParticlePerFrame<T_Pusher, MappingDesc::SuperCellSize, InterpolationScheme>;

        DataConnector& dc = Environment<>::get().DataConnector();
        auto fieldE = dc.get<FieldE>(FieldE::getName());
        auto fieldB = dc.get<FieldB>(FieldB::getName());

        /* Adjust interpolation area in particle pusher to allow sub-stepping pushes.
         * Here were provide an actual pusher and use its actual margins
         */
        using LowerMargin = typename GetLowerMarginForPusher<Particles, T_Pusher>::type;
        using UpperMargin = typename GetUpperMarginForPusher<Particles, T_Pusher>::type;

        using BlockArea = SuperCellDescription<typename MappingDesc::SuperCellSize, LowerMargin, UpperMargin>;

        auto const mapper = makeAreaMapper<CORE + BORDER>(this->cellDescription);

        auto workerCfg = pmacc::lockstep::makeWorkerCfg(*this);

        PMACC_LOCKSTEP_KERNEL(KernelMoveAndMarkParticles<BlockArea>{}, workerCfg)
        (mapper.getGridDim())(
            this->getDeviceParticlesBox(),
            fieldE->getDeviceDataBox(),
            fieldB->getDeviceDataBox(),
            currentStep,
            FrameSolver(),
            mapper);

        // The move-and-mark kernel sets mustShift for supercells, so we can call the optimized version of shift
        auto const onlyProcessMustShiftSupercells = true;
        shiftBetweenSupercells(pmacc::AreaMapperFactory<CORE + BORDER>{}, onlyProcessMustShiftSupercells);
    }

    template<typename T_Name, typename T_Flags, typename T_Attributes>
    template<typename T_MapperFactory>
    void Particles<T_Name, T_Flags, T_Attributes>::shiftBetweenSupercells(
        T_MapperFactory const& mapperFactory,
        bool const onlyProcessMustShiftSupercells)
    {
        ParticlesBaseType::template shiftParticles(mapperFactory, onlyProcessMustShiftSupercells);
    }

    template<typename T_Name, typename T_Flags, typename T_Attributes>
    template<typename T_DensityFunctor, typename T_PositionFunctor>
    void Particles<T_Name, T_Flags, T_Attributes>::initDensityProfile(
        T_DensityFunctor& densityFunctor,
        T_PositionFunctor& positionFunctor,
        const uint32_t currentStep)
    {
        log<picLog::SIMULATION_STATE>("initialize density profile for species %1%") % FrameType::getName();

        uint32_t const numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
        SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
        DataSpace<simDim> localCells = subGrid.getLocalDomain().size;
        DataSpace<simDim> totalGpuCellOffset = subGrid.getLocalDomain().offset;
        totalGpuCellOffset.y() += numSlides * localCells.y();

        auto const mapper = makeAreaMapper<CORE + BORDER>(this->cellDescription);
        auto workerCfg = lockstep::makeWorkerCfg(SuperCellSize{});
        PMACC_LOCKSTEP_KERNEL(KernelFillGridWithParticles<Particles>{}, workerCfg)
        (mapper.getGridDim())(
            densityFunctor,
            positionFunctor,
            totalGpuCellOffset,
            this->particlesBuffer->getDeviceParticleBox(),
            mapper);

        this->fillAllGaps();
    }

    template<typename T_Name, typename T_Flags, typename T_Attributes>
    template<
        typename T_SrcName,
        typename T_SrcAttributes,
        typename T_SrcFlags,
        typename T_ManipulateFunctor,
        typename T_SrcFilterFunctor>
    void Particles<T_Name, T_Flags, T_Attributes>::deviceDeriveFrom(
        Particles<T_SrcName, T_SrcAttributes, T_SrcFlags>& src,
        T_ManipulateFunctor& manipulatorFunctor,
        T_SrcFilterFunctor& srcFilterFunctor)
    {
        log<picLog::SIMULATION_STATE>("clone species %1%") % FrameType::getName();

        auto const mapper = makeAreaMapper<CORE + BORDER>(this->cellDescription);

        auto workerCfg = lockstep::makeWorkerCfg(*this);

        PMACC_LOCKSTEP_KERNEL(KernelDeriveParticles{}, workerCfg)
        (mapper.getGridDim())(
            this->getDeviceParticlesBox(),
            src.getDeviceParticlesBox(),
            manipulatorFunctor,
            srcFilterFunctor,
            mapper);
        this->fillAllGaps();
    }

} // namespace picongpu
