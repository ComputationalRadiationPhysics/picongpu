/* Copyright 2023-2024 Tapish Narwal
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

// required becuase the definition of Binner is conditionally included
#if(ENABLE_OPENPMD == 1)

#    include "picongpu/plugins/binning/BinningData.hpp"
#    include "picongpu/plugins/binning/BinningFunctors.hpp"
#    include "picongpu/plugins/binning/binners/Binner.hpp"
#    include "picongpu/plugins/binning/utility.hpp"
#    include "picongpu/plugins/misc/ExecuteIf.hpp"

#    include <pmacc/meta/errorHandlerPolicies/ReturnType.hpp>
#    include <pmacc/mpi/MPIReduce.hpp>
#    include <pmacc/mpi/reduceMethods/Reduce.hpp>

#    include <cstdint>


namespace picongpu
{
    namespace plugins::binning
    {
        template<typename TBinningData>
        class ParticleBinner : public Binner<TBinningData>
        {
        public:
            ParticleBinner(TBinningData const& bd, MappingDesc* cellDesc) : Binner<TBinningData>(bd, cellDesc)
            {
            }


            /**
             * onParticleLeave is called every time step whenever particles leave, it is independent of the notify
             * period. onParticleLeave isnt called for timestep 0, whereas notify is. Even though it is called every
             * timestep, notify must still be correctly set up for normalization, averaging and output. If binning only
             * leaving particles, use notify starting from 1 if you use time averaging, otherwise you have an extra
             * accumulate count at 0, when notify is called but onParticleLeave isnt.
             */
            void onParticleLeave(const std::string& speciesName, int32_t const direction) override
            {
                if(this->binningData.notifyPeriod.empty())
                    return;

                if(this->binningData.isRegionEnabled(ParticleRegion::Leaving))
                {
                    std::apply(
                        [&](auto const&... tupleArgs)
                        {
                            (misc::ExecuteIf{}(
                                 std::bind(
                                     BinLeavingParticles<typename std::decay_t<decltype(tupleArgs)>>{},
                                     this,
                                     tupleArgs,
                                     direction),
                                 misc::SpeciesNameIsEqual<typename std::decay_t<decltype(tupleArgs)>::species_type>{},
                                 speciesName),
                             ...);
                        },
                        this->binningData.speciesTuple);
                }
            }

        private:
            void doBinning(uint32_t currentStep) override
            {
                //  Do binning for species. Writes to histBuffer
                if(this->binningData.isRegionEnabled(ParticleRegion::Bounded))
                {
                    std::apply(
                        [&](auto const&... tupleArgs) { ((doBinningForSpecies(tupleArgs, currentStep)), ...); },
                        this->binningData.speciesTuple);
                }
            }

            template<typename T_FilteredSpecies>
            void doBinningForSpecies(T_FilteredSpecies const& fs, uint32_t currentStep)
            {
                using Species = pmacc::particles::meta::
                    FindByNameOrType_t<VectorAllSpecies, typename T_FilteredSpecies::species_type>;

                DataConnector& dc = Environment<>::get().DataConnector();
                auto particles = dc.get<Species>(Species::FrameType::getName());

                // @todo do species filtering

                auto particlesBox = particles->getDeviceParticlesBox();
                auto binningBox = this->histBuffer->getDeviceBuffer().getDataBox();

                auto cellDesc = *this->cellDescription;
                auto const mapper = makeAreaMapper<pmacc::type::CORE + pmacc::type::BORDER>(cellDesc);

                auto const globalOffset = Environment<simDim>::get().SubGrid().getGlobalDomain().offset;
                auto const localOffset = Environment<simDim>::get().SubGrid().getLocalDomain().offset;

                auto const axisKernels = tupleMap(
                    this->binningData.axisTuple,
                    [&](auto const& axis) -> decltype(auto) { return axis.getAxisKernel(); });

                auto const extraData = std::apply(
                    [&](auto&&... extras)
                    { return pmacc::memory::tuple::make_tuple(std::forward<decltype(extras)>(extras)...); },
                    this->binningData.extraData);
                auto const functorBlock = BinningFunctor{};

                PMACC_LOCKSTEP_KERNEL(functorBlock)
                    .config(mapper.getGridDim(), particlesBox)(
                        binningBox,
                        particlesBox,
                        localOffset,
                        globalOffset,
                        axisKernels,
                        this->binningData.depositionData.functor,
                        this->binningData.axisExtentsND,
                        extraData,
                        fs.filter,
                        currentStep,
                        mapper);
            }

            template<typename T_FilteredSpecies>
            struct BinLeavingParticles
            {
                using Species = pmacc::particles::meta::FindByNameOrType_t<
                    VectorAllSpecies,
                    typename T_FilteredSpecies::species_type,
                    pmacc::errorHandlerPolicies::ReturnType<void>>;

                template<typename T_BinData>
                auto operator()(
                    [[maybe_unused]] ParticleBinner<T_BinData>* binner,
                    T_FilteredSpecies const& fs,
                    [[maybe_unused]] int32_t direction) const -> void
                {
                    if constexpr(!std::is_same_v<void, Species>)
                    {
                        auto& dc = Environment<>::get().DataConnector();
                        auto particles = dc.get<Species>(Species::FrameType::getName());
                        auto particlesBox = particles->getDeviceParticlesBox();
                        auto binningBox = binner->histBuffer->getDeviceBuffer().getDataBox();

                        auto mapperFactory = particles::boundary::getMapperFactory(*particles, direction);
                        auto const mapper = mapperFactory(*(binner->cellDescription));

                        auto const globalOffset = Environment<simDim>::get().SubGrid().getGlobalDomain().offset;
                        auto const localOffset = Environment<simDim>::get().SubGrid().getLocalDomain().offset;

                        auto const axisKernels = tupleMap(
                            binner->binningData.axisTuple,
                            [&](auto const& axis) -> decltype(auto) { return axis.getAxisKernel(); });

                        pmacc::DataSpace<simDim> beginExternalCellsTotal, endExternalCellsTotal;
                        particles::boundary::getExternalCellsTotal(
                            *particles,
                            direction,
                            &beginExternalCellsTotal,
                            &endExternalCellsTotal);

                        auto const shiftTotaltoLocal = globalOffset + localOffset;
                        auto const beginExternalCellsLocal = beginExternalCellsTotal - shiftTotaltoLocal;
                        auto const endExternalCellsLocal = endExternalCellsTotal - shiftTotaltoLocal;

                        auto const extraData = std::apply(
                            [&](auto&&... extras)
                            { return pmacc::memory::tuple::make_tuple(std::forward<decltype(extras)>(extras)...); },
                            binner->binningData.extraData);

                        auto const functorLeaving = BinningFunctorLeaving{};

                        PMACC_LOCKSTEP_KERNEL(functorLeaving)
                            .config(mapper.getGridDim(), particlesBox)(
                                binningBox,
                                particlesBox,
                                localOffset,
                                globalOffset,
                                axisKernels,
                                binner->binningData.depositionData.functor,
                                binner->binningData.axisExtentsND,
                                extraData,
                                fs.filter,
                                Environment<>::get().SimulationDescription().getCurrentStep(),
                                beginExternalCellsLocal,
                                endExternalCellsLocal,
                                mapper);
                    }
                }
            };
        };
    } // namespace plugins::binning
} // namespace picongpu

#endif
