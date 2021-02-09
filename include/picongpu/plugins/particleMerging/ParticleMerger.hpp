/* Copyright 2017-2021 Heiko Burau
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

#include "picongpu/plugins/particleMerging/ParticleMerger.kernel"

#include "picongpu/simulation_defines.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"

#include <pmacc/traits/HasIdentifier.hpp>
#include <pmacc/cuSTL/cursor/MultiIndexCursor.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>

namespace picongpu
{
    namespace plugins
    {
        namespace particleMerging
        {
            using namespace pmacc;
            namespace bmpl = boost::mpl;

            /** Implements a particle merging algorithm based on
             *
             * Luu, P. T., Tueckmantel, T., & Pukhov, A. (2016).
             * Voronoi particle merging algorithm for PIC codes.
             * Computer Physics Communications, 202, 165-174.
             *
             * \tparam T_ParticlesType particle species
             */
            template<
                class T_ParticlesType,
                bool hasVoronoiCellId
                = pmacc::traits::HasIdentifier<typename T_ParticlesType::FrameType, voronoiCellId>::type::value>
            struct ParticleMergerWrapped;


            template<class T_ParticlesType>
            struct ParticleMergerWrapped<T_ParticlesType, true> : ISimulationPlugin
            {
            private:
                std::string name;
                std::string prefix;
                std::string notifyPeriod;
                MappingDesc* cellDescription;

                uint32_t minParticlesToMerge;
                float_X posSpreadThreshold;
                float_X absMomSpreadThreshold_mc;
                float_X absMomSpreadThreshold;
                float_X relMomSpreadThreshold;
                float_64 minMeanEnergy_keV;
                float_X minMeanEnergy;

            public:
                using ParticlesType = T_ParticlesType;

                ParticleMergerWrapped()
                    : name("ParticleMerger: merges several macroparticles with"
                           " similar position and momentum into a single one")
                    , prefix(ParticlesType::FrameType::getName() + std::string("_merger"))
                    , cellDescription(nullptr)
                {
                    Environment<>::get().PluginConnector().registerPlugin(this);
                }

                void notify(uint32_t currentStep)
                {
                    using SuperCellSize = MappingDesc::SuperCellSize;

                    const pmacc::math::Int<simDim> coreBorderGuardSuperCells
                        = this->cellDescription->getGridSuperCells();
                    const pmacc::math::Int<simDim> guardSuperCells = this->cellDescription->getGuardingSuperCells();
                    const pmacc::math::Int<simDim> coreBorderSuperCells
                        = coreBorderGuardSuperCells - 2 * guardSuperCells;

                    /* this zone represents the core+border area with guard offset in unit of cells */
                    const zone::SphericZone<simDim> zone(
                        static_cast<pmacc::math::Size_t<simDim>>(coreBorderSuperCells * SuperCellSize::toRT()),
                        guardSuperCells * SuperCellSize::toRT());

                    /* get particles instance */
                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName(), true);

                    /* create `ParticleMergerKernel` instance */
                    ParticleMergerKernel<typename ParticlesType::ParticlesBoxType> particleMergerKernel(
                        particles->getDeviceParticlesBox(),
                        this->minParticlesToMerge,
                        this->posSpreadThreshold,
                        this->absMomSpreadThreshold,
                        this->relMomSpreadThreshold,
                        this->minMeanEnergy);

                    /* execute particle merging alorithm */
                    algorithm::kernel::Foreach<SuperCellSize> foreach;
                    foreach(zone, cursor::make_MultiIndexCursor<simDim>(), particleMergerKernel)
                        ;

                    /* close all gaps caused by removal of particles */
                    particles->fillAllGaps();
                }


                void setMappingDescription(MappingDesc* cellDescription)
                {
                    this->cellDescription = cellDescription;
                }


                void pluginRegisterHelp(po::options_description& desc)
                {
                    desc.add_options()(
                        (this->prefix + ".period").c_str(),
                        po::value<std::string>(&this->notifyPeriod),
                        "enable plugin [for each n-th step]")(
                        (this->prefix + ".minParticlesToMerge").c_str(),
                        po::value<uint32_t>(&this->minParticlesToMerge)->default_value(8),
                        "minimal number of macroparticles needed to merge"
                        " the macroparticle collection into a single macroparticle.")(
                        (this->prefix + ".posSpreadThreshold").c_str(),
                        po::value<float_X>(&this->posSpreadThreshold)->default_value(0.5),
                        "Below this threshold of spread in position macroparticles"
                        " can be merged [unit: cell edge length].")(
                        (this->prefix + ".absMomSpreadThreshold").c_str(),
                        po::value<float_X>(&this->absMomSpreadThreshold_mc)->default_value(-1.0),
                        "Below this absolute threshold of spread in momentum"
                        " macroparticles can be merged [unit: m_el * c]."
                        " Disabled for -1 (default).")(
                        (this->prefix + ".relMomSpreadThreshold").c_str(),
                        po::value<float_X>(&this->relMomSpreadThreshold)->default_value(-1.0),
                        "Below this relative (to mean momentum) threshold of spread in"
                        " momentum macroparticles can be merged [unit: none]."
                        " Disabled for -1 (default).")(
                        (this->prefix + ".minMeanEnergy").c_str(),
                        po::value<float_64>(&this->minMeanEnergy_keV)->default_value(511.0),
                        "minimal mean kinetic energy needed to merge the macroparticle"
                        " collection into a single macroparticle [unit: keV].");
                }

                std::string pluginGetName() const
                {
                    return this->name;
                }

            protected:
                void pluginLoad()
                {
                    if(notifyPeriod.empty())
                        return;

                    Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);

                    // clean user parameters
                    PMACC_VERIFY_MSG(
                        this->minParticlesToMerge > 1,
                        std::string("[Plugin: ") + this->prefix
                            + "] minParticlesToMerge"
                              " has to be greater than one.");
                    PMACC_VERIFY_MSG(
                        this->posSpreadThreshold >= float_X(0.0),
                        std::string("[Plugin: ") + this->prefix
                            + "] posSpreadThreshold"
                              " has to be non-negative.");
                    PMACC_VERIFY_MSG(
                        this->absMomSpreadThreshold_mc * this->relMomSpreadThreshold < float(0.0),
                        std::string("[Plugin: ") + this->prefix
                            + "] either"
                              " absMomSpreadThreshold or relMomSpreadThreshold has to be given");
                    PMACC_VERIFY_MSG(
                        this->minMeanEnergy >= float_X(0.0),
                        std::string("[Plugin: ") + this->prefix
                            + "] minMeanEnergy"
                              " has to be non-negative.");

                    // convert units of user parameters
                    this->absMomSpreadThreshold = this->absMomSpreadThreshold_mc * ELECTRON_MASS * SPEED_OF_LIGHT;

                    const float_64 minMeanEnergy_SI = this->minMeanEnergy_keV * UNITCONV_keV_to_Joule;
                    this->minMeanEnergy = static_cast<float_X>(minMeanEnergy_SI / UNIT_ENERGY);
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


            template<class T_ParticlesType>
            struct ParticleMergerWrapped<T_ParticlesType, false> : ISimulationPlugin
            {
            private:
                std::string name;
                std::string prefix;
                std::string notifyPeriod;
                MappingDesc* cellDescription;

            public:
                using ParticlesType = T_ParticlesType;

                ParticleMergerWrapped()
                    : name("ParticleMerger: merges several macroparticles with"
                           " similar position and momentum into a single one.\n"
                           "plugin disabled. Enable plugin by adding the `voronoiCellId`"
                           " attribute to the particle attribute list.")
                    , prefix(ParticlesType::FrameType::getName() + std::string("_merger"))
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


            template<typename T_ParticlesType>
            struct ParticleMerger : ParticleMergerWrapped<T_ParticlesType>
            {
            };

        } // namespace particleMerging
    } // namespace plugins
} // namespace picongpu
