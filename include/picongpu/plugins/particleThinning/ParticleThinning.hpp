/* Copyright 2020 Xeinia Bastrakova, Sergei Bastrakov
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

#include "reduction_library/thinning/InKernelThinning.hpp"
#include "picongpu/plugins/particleThinning/ParticleThinning.kernel"


namespace picongpu
{
namespace plugins
{
namespace particleThinning
{

    using namespace pmacc;
    namespace bmpl = boost::mpl;

    template< class T_ParticlesType >
    struct ParticleThinning : ISimulationPlugin
    {
    private:
        std::string name;
        std::string prefix;
        std::string notifyPeriod;
        MappingDesc* cellDescription;

        float_X ratioDeletedParticles;

    public:
        using ParticlesType = T_ParticlesType;

        ParticleThinning() :
            name(
                "ParticleThinning: thins out macroparticles"
            ),
            prefix( ParticlesType::FrameType::getName() + std::string("_particleThinning") ),
            cellDescription( nullptr )
        {
            Environment<>::get().PluginConnector().registerPlugin( this );
        }

        void notify(uint32_t currentStep) override
        {
            DataConnector &dc = Environment<>::get().DataConnector();
            auto particles = dc.get< ParticlesType >(
                ParticlesType::FrameType::getName(),
                true
            );

            const pmacc::math::Int<simDim> guardSuperCells =
                this->cellDescription->getGuardingSuperCells();


            AreaMapping<
                CORE + BORDER, // full local domain, no guards
                MappingDesc
            > mapper( *cellDescription );
            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                pmacc::math::CT::volume< MappingDesc::SuperCellSize >::type::value
            >::value;

            // call a kernel
            auto kernel = ParticleThinningKernel<
                numWorkers
            >{ };

            using namespace pmacc::random::distributions;
            using Distribution = Uniform<float_X>;
            using RngFactory = particles::functor::misc::Rng< Distribution >;
            RngFactory rngFactory( currentStep );

            PMACC_KERNEL( kernel )(
                mapper.getGridDim(), // how many blocks = how many supercells in local domain
                numWorkers           // how many threads per block
            )(
                particles->getDeviceParticlesBox( ),
                mapper,
                ratioDeletedParticles,
                rngFactory,
                guardSuperCells
            );

            // close all gaps caused by removal of particles
            particles->fillAllGaps();
        }


        void setMappingDescription(MappingDesc* cellDescription) override
        {
            this->cellDescription = cellDescription;
        }


        void pluginRegisterHelp(po::options_description& desc) override
        {
            desc.add_options()
            (
                ( prefix + ".period" ).c_str(),
                po::value< std::string > (
                    &notifyPeriod
                ),
                "enable plugin [for each n-th step]"
            )
            (
                ( prefix + ".ratioDeletedParticles" ).c_str(),
                po::value< float_X > (
                    &ratioDeletedParticles
                )->default_value( 0.1 ),
                "Ratio of macroparticles to be deleted on average"
            );
        }

        std::string pluginGetName() const override
        {
            return name;
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
                ratioDeletedParticles > 0.0_X,
                std::string("[Plugin: ") + prefix + "] ratioDeletedParticles"
                " has to be > 0."
            );
            PMACC_VERIFY_MSG(
                ratioDeletedParticles < 1.0_X,
                std::string("[Plugin: ") + prefix + "] ratioDeletedParticles"
                " has to be < 1."
            );
        }

        void pluginUnload()
        {}

        void restart( uint32_t, const std::string )
        {}

        void checkpoint( uint32_t, const std::string )
        {}
    };

} // namespace particleThinning
} // namespace plugins
} // namespace picongpu
