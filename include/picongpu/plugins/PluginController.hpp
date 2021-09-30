/* Copyright 2013-2022 Axel Huebl, Benjamin Schneider, Felix Schmitt,
 *                     Heiko Burau, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz, Erik Zenker, Finn-Ole Carstens,
 *                     Franz Poeschel
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

#include "picongpu/plugins/BinEnergyParticles.hpp"
#include "picongpu/plugins/CountParticles.hpp"
#include "picongpu/plugins/Emittance.hpp"
#include "picongpu/plugins/EnergyFields.hpp"
#include "picongpu/plugins/EnergyParticles.hpp"
#include "picongpu/plugins/SumCurrents.hpp"
#include "picongpu/plugins/multi/Master.hpp"
#include "picongpu/plugins/output/images/PngCreator.hpp"
#include "picongpu/plugins/output/images/Visualisation.hpp"
#include "picongpu/plugins/transitionRadiation/TransitionRadiation.hpp"

#include <pmacc/assert.hpp>
/* That's an abstract plugin for image output with the possibility
 * to store the image as png file or send it via a sockets to a server.
 *
 * \todo rename PngPlugin to ImagePlugin or similar
 */
#include "picongpu/plugins/PngPlugin.hpp"

#if(ENABLE_OPENPMD == 1)
#    include "picongpu/plugins/PhaseSpace/PhaseSpace.hpp"
#    include "picongpu/plugins/openPMD/openPMDWriter.hpp"
#    include "picongpu/plugins/particleCalorimeter/ParticleCalorimeter.hpp"
#endif

#include "picongpu/plugins/ChargeConservation.hpp"
#if(ENABLE_OPENPMD == 1)
#    include "picongpu/plugins/makroParticleCounter/PerSuperCell.hpp"
#endif

#if(ENABLE_ISAAC == 1) && (SIMDIM == DIM3)
#    include "picongpu/plugins/IsaacPlugin.hpp"
#endif

#if ENABLE_OPENPMD
#    include "picongpu/plugins/radiation/Radiation.hpp"
#    include "picongpu/plugins/radiation/VectorTypes.hpp"
#endif

#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/Checkpoint.hpp"
#include "picongpu/plugins/ILightweightPlugin.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"

#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/meta/AllCombinations.hpp>
#include <pmacc/meta/SeqToList.hpp>

#include <list>
#include <memory>


namespace picongpu
{
    using namespace pmacc;

    /**
     * Plugin management controller for user-level plugins.
     */
    class PluginController : public ILightweightPlugin
    {
    private:
        std::list<std::shared_ptr<ISimulationPlugin>> plugins;

        template<typename T_Type>
        struct PushBack
        {
            template<typename T>
            void operator()(T& list)
            {
                list.push_back(std::make_shared<T_Type>());
            }
        };

        struct TupleSpeciesPlugin
        {
            enum Names
            {
                species = 0,
                plugin = 1
            };

            /** apply the 1st vector component to the 2nd
             *
             * @tparam T_TupleVector vector of type
             *                       pmacc::math::CT::vector< Species, Plugin >
             *                       with two components
             */
            template<typename T_TupleVector>
            using Apply = typename boost::mpl::
                apply1<pmacc::mp_at_c<T_TupleVector, plugin>, pmacc::mp_at_c<T_TupleVector, species>>::type;

            /** Check the combination Species+Plugin in the Tuple
             *
             * @tparam T_TupleVector with Species, Plugin
             */
            template<typename T_TupleVector>
            struct IsEligible
            {
                using Species = pmacc::mp_at_c<T_TupleVector, species>;
                using Solver = pmacc::mp_at_c<T_TupleVector, plugin>;

                static constexpr bool value
                    = particles::traits::SpeciesEligibleForSolver<Species, Solver>::type::value;
            };
        };

        /* define stand alone plugins */
        using StandAlonePlugins = pmacc::mp_list<
            Checkpoint,
            EnergyFields,
            ChargeConservation

#if(ENABLE_OPENPMD == 1)
            ,
            plugins::multi::Master<openPMD::openPMDWriter>
#endif

#if(PMACC_CUDA_ENABLED == 1)
            ,
            SumCurrents
#endif

#if(ENABLE_ISAAC == 1) && (SIMDIM == DIM3)
            ,
            isaacP::IsaacPlugin
#endif
            >;

        using AllFields = pmacc::mp_list<FieldB, FieldE, FieldJ>;

        /* define species plugins */
        using UnspecializedSpeciesPlugins = pmacc::mp_list<
            plugins::multi::Master<EnergyParticles<boost::mpl::_1>>,
            plugins::multi::Master<CalcEmittance<boost::mpl::_1>>,
            plugins::multi::Master<BinEnergyParticles<boost::mpl::_1>>,
            CountParticles<boost::mpl::_1>,
            PngPlugin<Visualisation<boost::mpl::_1, PngCreator>>,
            plugins::transitionRadiation::TransitionRadiation<boost::mpl::_1>
#if ENABLE_OPENPMD
            ,
            plugins::radiation::Radiation<boost::mpl::_1>
#endif
#if(ENABLE_OPENPMD == 1)
            ,
            plugins::multi::Master<ParticleCalorimeter<boost::mpl::_1>>,
            plugins::multi::Master<PhaseSpace<particles::shapes::Counter::ChargeAssignment, boost::mpl::_1>>
#endif
#if(ENABLE_OPENPMD == 1)
            ,
            PerSuperCell<boost::mpl::_1>
#endif
            >;

        using CombinedUnspecializedSpeciesPlugins
            = pmacc::AllCombinations<VectorAllSpecies, UnspecializedSpeciesPlugins>;

        using CombinedUnspecializedSpeciesPluginsEligible
            = pmacc::mp_copy_if<CombinedUnspecializedSpeciesPlugins, TupleSpeciesPlugin::IsEligible>;

        using SpeciesPlugins
            = pmacc::mp_transform<TupleSpeciesPlugin::Apply, CombinedUnspecializedSpeciesPluginsEligible>;

        /* create sequence with all fully specialized plugins */
        using AllPlugins = MakeSeq_t<StandAlonePlugins, SpeciesPlugins>;

        /**
         * Initializes the controller by adding all user plugins to its internal list.
         */
        virtual void init()
        {
            meta::ForEach<AllPlugins, PushBack<boost::mpl::_1>> pushBack;
            pushBack(plugins);
        }

    public:
        PluginController()
        {
            init();
        }

        ~PluginController() override = default;

        void setMappingDescription(MappingDesc* cellDescription) override
        {
            PMACC_ASSERT(cellDescription != nullptr);

            for(auto iter = plugins.begin(); iter != plugins.end(); ++iter)
            {
                (*iter)->setMappingDescription(cellDescription);
            }
        }

        void pluginRegisterHelp(po::options_description&) override
        {
            // no help required at the moment
        }

        std::string pluginGetName() const override
        {
            return "PluginController";
        }

        void notify(uint32_t) override
        {
        }

        void pluginUnload() override
        {
            plugins.clear();
        }
    };

} // namespace picongpu
