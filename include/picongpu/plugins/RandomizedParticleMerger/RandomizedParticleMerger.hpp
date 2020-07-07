/* Copyright 2017-2020 Heiko Burau
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
#include "picongpu/particles/functor/misc/Rng.hpp"

#include <pmacc/traits/HasIdentifier.hpp>
#include <pmacc/cuSTL/cursor/MultiIndexCursor.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "RandomizedParticleMerger.kernel"

namespace picongpu
{
namespace plugins
{
namespace randomizedParticleMerger
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
        bool hasVoronoiCellId =
            pmacc::traits::HasIdentifier<
                typename T_ParticlesType::FrameType,
                voronoiCellId
            >::type::value
    >
    struct RandomizedParticleMergerWrapped;


    template< class T_ParticlesType >
    struct RandomizedParticleMergerWrapped< T_ParticlesType, true > : ISimulationPlugin
    {
    private:
        std::string name;
        std::string prefix;
        std::string notifyPeriod;
        MappingDesc* cellDescription;

        uint32_t minMacroParticlesToDivide;
        float_X ratioDeletedParticles;
        float_X posSpreadThreshold;
        float_X momSpreadThreshold;

    public:
        using ParticlesType = T_ParticlesType;

        RandomizedParticleMergerWrapped() :
            name(
                "ParticleMerger: merges several macroparticles with"
                " similar position and momentum into a single one"
            ),
            prefix( ParticlesType::FrameType::getName() + std::string("_randomizedMerger") ),
            cellDescription( nullptr )
        {
            Environment<>::get().PluginConnector().registerPlugin( this );
        }

        void notify(uint32_t currentStep)
        {
            using SuperCellSize = MappingDesc::SuperCellSize;

            const pmacc::math::Int<simDim> coreBorderGuardSuperCells =
                this->cellDescription->getGridSuperCells();
            const pmacc::math::Int<simDim> guardSuperCells =
                this->cellDescription->getGuardingSuperCells();
            const pmacc::math::Int<simDim> coreBorderSuperCells =
                coreBorderGuardSuperCells - 2 * guardSuperCells;

            /* this zone represents the core+border area with guard offset in unit of cells */
            const zone::SphericZone< simDim > zone(
                static_cast< pmacc::math::Size_t< simDim > >(
                    coreBorderSuperCells * SuperCellSize::toRT()
                ),
                guardSuperCells * SuperCellSize::toRT()
            );

            /* get particles instance */
            DataConnector &dc = Environment<>::get().DataConnector();
            auto particles = dc.get< ParticlesType >(
                ParticlesType::FrameType::getName(),
                true
            );
            /* making fabric for random value*/
            using namespace pmacc::random::distributions;
            using Distribution = Uniform<float>;
            using RngFactory = particles::functor::misc::Rng< Distribution >;
            RngFactory rngFactory(currentStep);

            /* create `RandomizeParticleMergerKernel` instance */
            RandomizedParticleMergerKernel< typename ParticlesType::ParticlesBoxType >
            randomizedParticleMergerKernel(
                particles->getDeviceParticlesBox(),
                this->minMacroParticlesToDivide,
                this->ratioDeletedParticles,
                this->posSpreadThreshold,
                this->momSpreadThreshold,
                rngFactory,
                guardSuperCells
            );

            /* execute particle merging alorithm */
            algorithm::kernel::Foreach< SuperCellSize > foreach;
            foreach(
                zone,
                cursor::make_MultiIndexCursor< simDim >(),
                randomizedParticleMergerKernel
            );

            /* close all gaps caused by removal of particles */
            particles->fillAllGaps();

        }


        void setMappingDescription(MappingDesc* cellDescription)
        {
            this->cellDescription = cellDescription;
        }


        void pluginRegisterHelp(po::options_description& desc)
        {

            desc.add_options()
            (
                ( this->prefix + ".period" ).c_str(),
                po::value< std::string > (
                    &this->notifyPeriod
                ),
                "enable plugin [for each n-th step]"
            )
            (
                ( this->prefix + ".minMacroParticlesToDivide" ).c_str(),
                po::value< uint32_t > (
                    &this->minMacroParticlesToDivide
                )->default_value( 8 ),
                "minimum number of macro particles at which we always divide the cell"
            )
            (
                ( this->prefix + ".posSpreadThreshold" ).c_str(),
                po::value< float_X > (
                    &this->posSpreadThreshold
                )->default_value( 1e-5 ),
                "Below this threshold of spread in position macroparticles"
                " can be merged [unit: cell edge length]."
             )
            (
                ( this->prefix + ".momSpreadThreshold" ).c_str(),
                po::value< float_X > (
                    &this->momSpreadThreshold
                )->default_value( 1e-5 ),
                "Below this absolute threshold of spread in momentum"
                " macroparticles can be merged [unit: m_el * c]."
                " Disabled for -1 (default)."
            )
             (
                ( this->prefix + ".ratioDeletedParticles" ).c_str(),
                po::value< float_X > (
                    &this->ratioDeletedParticles
                )->default_value( 0.1 ),
                "Ratio of deleted particle"
            );
        }

        std::string pluginGetName() const
        {
            return this->name;
        }

    protected:

        void pluginLoad()
        {
            if( notifyPeriod.empty() )
               return;

            Environment<>::get().PluginConnector().setNotificationPeriod(
                this,
                notifyPeriod
            );

            PMACC_VERIFY_MSG(
                this->minMacroParticlesToDivide > 1,
                std::string("[Plugin: ") + this->prefix + "] minMacroParticlesToDivide"
                " has to be greater than one."
            );
            PMACC_VERIFY_MSG(
                this->ratioDeletedParticles >= float_X(0.0),
                std::string("[Plugin: ") + this->prefix + "] ratioDeletedParticles"
                " has to be non-negative."
            );
            PMACC_VERIFY_MSG(
                this->posSpreadThreshold >= float_X(0.0),
                std::string("[Plugin: ") + this->prefix + "] posSpreadThreshold"
                " has to be non-negative."
            );
            PMACC_VERIFY_MSG(
                this->momSpreadThreshold >= float_X(0.0),
                std::string("[Plugin: ") + this->prefix + "] momSpreadThreshold"
                " has to be non-negative."
            );
        }

        void pluginUnload()
        {}

        void restart( uint32_t, const std::string )
        {}

        void checkpoint( uint32_t, const std::string )
        {}
    };


    template< class T_ParticlesType >
    struct RandomizedParticleMergerWrapped< T_ParticlesType, false > : ISimulationPlugin
    {
    private:
        std::string name;
        std::string prefix;
        std::string notifyPeriod;
        MappingDesc* cellDescription;

    public:
        using ParticlesType = T_ParticlesType;

        RandomizedParticleMergerWrapped() :
            name(
                "ParticleMerger: merges several macroparticles with"
                " similar position and momentum into a single one.\n"
                "plugin disabled. Enable plugin by adding the `voronoiCellId`"
                " attribute to the particle attribute list."
            ),
            prefix( ParticlesType::FrameType::getName() + std::string("_randomizedMerger") ),
            cellDescription( nullptr )
        {
            Environment<>::get().PluginConnector().registerPlugin( this );
        }

        std::string pluginGetName() const
        {
            return this->name;
        }

    protected:
        void setMappingDescription( MappingDesc* )
        {}

        void pluginRegisterHelp( po::options_description& )
        {}

        void pluginUnload()
        {}

        void restart( uint32_t, const std::string )
        {}

        void checkpoint( uint32_t, const std::string )
        {}

        void notify( uint32_t )
        {}
    };


    template< typename T_ParticlesType >
    struct RandomizedParticleMerger : RandomizedParticleMergerWrapped< T_ParticlesType >
    {};

} // namespace randomizedParticleMerger
} // namespace plugins
} // namespace picongpu


