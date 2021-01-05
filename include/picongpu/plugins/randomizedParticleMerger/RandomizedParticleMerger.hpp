/* Copyright 2017-2021 Heiko Burau, Xeinia Bastrakova, Sergei Bastrakov
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
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/randomizedParticleMerger/RandomizedParticleMerger.kernel"
#include "picongpu/particles/functor/misc/Rng.hpp"

#include <pmacc/traits/HasIdentifier.hpp>
#include <pmacc/cuSTL/cursor/MultiIndexCursor.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>


namespace picongpu
{
    namespace plugins
    {
        namespace randomizedParticleMerger
        {
            using namespace pmacc;
            namespace bmpl = boost::mpl;

            /** Implements a randomized modification of the particle merging algorithm.
             *
             * The original particle merging algorithms is
             * Luu, P. T., Tueckmantel, T., & Pukhov, A. (2016).
             * Voronoi particle merging algorithm for PIC codes.
             * Computer Physics Communications, 202, 165-174.
             *
             * The randomized mofidication developed by S. Bastrakov and X. Bastrakova
             *
             * @tparam T_ParticlesType species type
             * @tparam hasVoronoiCellId if the species type has the voronoiCellId attribute,
             *                          the plugin will only be used for such types
             */
            template<
                class T_ParticlesType,
                bool hasVoronoiCellId
                = pmacc::traits::HasIdentifier<typename T_ParticlesType::FrameType, voronoiCellId>::type::value>
            struct RandomizedParticleMergerWrapped;

            template<class T_ParticlesType>
            struct RandomizedParticleMergerWrapped<T_ParticlesType, true> : ISimulationPlugin
            {
            private:
                std::string name;
                std::string prefix;
                std::string notifyPeriod;
                MappingDesc* cellDescription;

                uint32_t maxParticlesToMerge;
                float_X ratioDeletedParticles;
                float_X posSpreadThreshold;
                float_X momSpreadThreshold;

            public:
                using ParticlesType = T_ParticlesType;

                RandomizedParticleMergerWrapped()
                    : name("RandomizedParticleMerger: merges several macroparticles with"
                           " similar position and momentum into a single one")
                    , prefix(ParticlesType::FrameType::getName() + std::string("_randomizedMerger"))
                    , cellDescription(nullptr)
                {
                    Environment<>::get().PluginConnector().registerPlugin(this);
                }

                void notify(uint32_t currentStep) override
                {
                    using SuperCellSize = MappingDesc::SuperCellSize;

                    const pmacc::math::Int<simDim> coreBorderGuardSuperCells
                        = this->cellDescription->getGridSuperCells();
                    const pmacc::math::Int<simDim> guardSuperCells = this->cellDescription->getGuardingSuperCells();
                    const pmacc::math::Int<simDim> coreBorderSuperCells
                        = coreBorderGuardSuperCells - 2 * guardSuperCells;

                    // this zone represents the core+border area with guard offset in unit of cells
                    const zone::SphericZone<simDim> zone(
                        static_cast<pmacc::math::Size_t<simDim>>(coreBorderSuperCells * SuperCellSize::toRT()),
                        guardSuperCells * SuperCellSize::toRT());

                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName(), true);
                    using Kernel = RandomizedParticleMergerKernel<typename ParticlesType::ParticlesBoxType>;

                    using namespace pmacc::random::distributions;
                    using Distribution = Uniform<float_X>;
                    using RngFactory = particles::functor::misc::Rng<Distribution>;

                    RngFactory rngFactory(currentStep);
                    auto kernel = Kernel{
                        particles->getDeviceParticlesBox(),
                        maxParticlesToMerge,
                        ratioDeletedParticles,
                        posSpreadThreshold,
                        momSpreadThreshold,
                        rngFactory,
                        guardSuperCells};

                    algorithm::kernel::Foreach<SuperCellSize> foreach;
                    foreach(zone, cursor::make_MultiIndexCursor<simDim>(), kernel)
                        ;

                    // close all gaps caused by removal of particles
                    particles->fillAllGaps();
                }


                void setMappingDescription(MappingDesc* cellDescription) override
                {
                    this->cellDescription = cellDescription;
                }


                void pluginRegisterHelp(po::options_description& desc) override
                {
                    desc.add_options()(
                        (prefix + ".period").c_str(),
                        po::value<std::string>(&notifyPeriod),
                        "enable plugin [for each n-th step]")(
                        (prefix + ".maxParticlesToMerge").c_str(),
                        po::value<uint32_t>(&maxParticlesToMerge)->default_value(8),
                        "minimum number of macroparticles at which we always divide the cell")(
                        (prefix + ".posSpreadThreshold").c_str(),
                        po::value<float_X>(&posSpreadThreshold)->default_value(1e-5),
                        "Below this threshold of spread in position macroparticles"
                        " can be merged [unit: cell edge length]")(
                        (prefix + ".momSpreadThreshold").c_str(),
                        po::value<float_X>(&momSpreadThreshold)->default_value(1e-5),
                        "Below this absolute threshold of spread in momentum"
                        " macroparticles can be merged [unit: m_el * c].")(
                        (prefix + ".ratioDeletedParticles").c_str(),
                        po::value<float_X>(&ratioDeletedParticles)->default_value(0.1),
                        "Ratio of macroparticles to be deleted on average");
                }

                std::string pluginGetName() const override
                {
                    return name;
                }

            protected:
                void pluginLoad()
                {
                    if(notifyPeriod.empty())
                        return;

                    Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);

                    PMACC_VERIFY_MSG(
                        maxParticlesToMerge > 1u,
                        std::string("[Plugin: ") + prefix
                            + "] maxParticlesToMerge"
                              " has to be greater than one.");
                    PMACC_VERIFY_MSG(
                        ratioDeletedParticles > 0.0_X,
                        std::string("[Plugin: ") + prefix
                            + "] ratioDeletedParticles"
                              " has to be > 0.");
                    PMACC_VERIFY_MSG(
                        ratioDeletedParticles < 1.0_X,
                        std::string("[Plugin: ") + prefix
                            + "] ratioDeletedParticles"
                              " has to be < 1.");
                    PMACC_VERIFY_MSG(
                        posSpreadThreshold >= 0.0_X,
                        std::string("[Plugin: ") + prefix
                            + "] posSpreadThreshold"
                              " has to be non-negative.");
                    PMACC_VERIFY_MSG(
                        momSpreadThreshold >= 0.0_X,
                        std::string("[Plugin: ") + prefix
                            + "] momSpreadThreshold"
                              " has to be non-negative.");
                }

                void pluginUnload()
                {
                }

                void restart(uint32_t, const std::string)
                {
                }

                void checkpoint(uint32_t, const std::string)
                {
                }
            };


            /** Placeholder implementation for species without the required conditions
             *
             * @tparam T_ParticlesType species type
             */
            template<class T_ParticlesType>
            struct RandomizedParticleMergerWrapped<T_ParticlesType, false> : ISimulationPlugin
            {
            private:
                std::string name;
                std::string prefix;
                std::string notifyPeriod;
                MappingDesc* cellDescription;

            public:
                using ParticlesType = T_ParticlesType;

                RandomizedParticleMergerWrapped()
                    : name("RandomizedParticleMerger: merges several macroparticles with"
                           " similar position and momentum into a single one.\n"
                           "plugin disabled. Enable plugin by adding the `voronoiCellId`"
                           " attribute to the particle attribute list.")
                    , prefix(ParticlesType::FrameType::getName() + std::string("_randomizedMerger"))
                    , cellDescription(nullptr)
                {
                    Environment<>::get().PluginConnector().registerPlugin(this);
                }

                std::string pluginGetName() const
                {
                    return this->name;
                }

            protected:
                void setMappingDescription(MappingDesc*)
                {
                }

                void pluginRegisterHelp(po::options_description&)
                {
                }

                void pluginUnload()
                {
                }

                void restart(uint32_t, const std::string)
                {
                }

                void checkpoint(uint32_t, const std::string)
                {
                }

                void notify(uint32_t)
                {
                }
            };

            /** Randomized particle merger plugin
             *
             * @tparam T_ParticlesType species type
             */
            template<typename T_ParticlesType>
            struct RandomizedParticleMerger : RandomizedParticleMergerWrapped<T_ParticlesType>
            {
            };

        } // namespace randomizedParticleMerger
    } // namespace plugins
} // namespace picongpu
