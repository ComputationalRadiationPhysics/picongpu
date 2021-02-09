/* Copyright 2013-2021 Axel Huebl, Benjamin Schneider, Felix Schmitt,
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
#include <pmacc/assert.hpp>

#include "picongpu/plugins/CountParticles.hpp"
#include "picongpu/plugins/EnergyParticles.hpp"
#include "picongpu/plugins/multi/Master.hpp"
#include "picongpu/plugins/EnergyFields.hpp"
#include "picongpu/plugins/SumCurrents.hpp"
#include "picongpu/plugins/BinEnergyParticles.hpp"
#include "picongpu/plugins/Emittance.hpp"
#include "picongpu/plugins/transitionRadiation/TransitionRadiation.hpp"
#include "picongpu/plugins/output/images/PngCreator.hpp"
#include "picongpu/plugins/output/images/Visualisation.hpp"
/* That's an abstract plugin for image output with the possibility
 * to store the image as png file or send it via a sockets to a server.
 *
 * \todo rename PngPlugin to ImagePlugin or similar
 */
#include "picongpu/plugins/PngPlugin.hpp"

#if(ENABLE_ADIOS == 1)
#    include "picongpu/plugins/adios/ADIOSWriter.hpp"
#endif

#if(ENABLE_OPENPMD == 1)
#    include "picongpu/plugins/openPMD/openPMDWriter.hpp"
#    include "picongpu/plugins/PhaseSpace/PhaseSpace.hpp"
#    include "picongpu/plugins/xrayScattering/XrayScattering.hpp"
#endif

#if(PMACC_CUDA_ENABLED == 1)
#    include "picongpu/plugins/PositionsParticles.hpp"
#    include "picongpu/plugins/ChargeConservation.hpp"
#    include "picongpu/plugins/particleMerging/ParticleMerger.hpp"
#    include "picongpu/plugins/randomizedParticleMerger/RandomizedParticleMerger.hpp"
#    if(ENABLE_HDF5 == 1)
#        include "picongpu/plugins/makroParticleCounter/PerSuperCell.hpp"
#    endif

#    include "picongpu/plugins/SliceFieldPrinterMulti.hpp"
#    if(SIMDIM == DIM3)
#        include "picongpu/plugins/IntensityPlugin.hpp"
#    endif
#endif

#if(ENABLE_ISAAC == 1) && (SIMDIM == DIM3)
#    include "picongpu/plugins/IsaacPlugin.hpp"
#endif

#if(ENABLE_HDF5 == 1)
#    include "picongpu/plugins/particleCalorimeter/ParticleCalorimeter.hpp"
#    include "picongpu/plugins/radiation/VectorTypes.hpp"
#    include "picongpu/plugins/radiation/Radiation.hpp"
#endif

#include "picongpu/plugins/Checkpoint.hpp"
#include "picongpu/plugins/ResourceLog.hpp"

#include <pmacc/mappings/kernel/MappingDescription.hpp>

#include "picongpu/plugins/ILightweightPlugin.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"

#include <list>


namespace picongpu
{
    using namespace pmacc;

    /**
     * Plugin management controller for user-level plugins.
     */
    class PluginController : public ILightweightPlugin
    {
    private:
        std::list<ISimulationPlugin*> plugins;

        template<typename T_Type>
        struct PushBack
        {
            template<typename T>
            void operator()(T& list)
            {
                list.push_back(new T_Type());
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
            struct Apply
                : bmpl::apply1<
                      typename pmacc::math::CT::At<T_TupleVector, bmpl::int_<plugin>>::type,
                      typename pmacc::math::CT::At<T_TupleVector, bmpl::int_<species>>::type>
            {
            };

            /** Check the combination Species+Plugin in the Tuple
             *
             * @tparam T_TupleVector with Species, Plugin
             */
            template<typename T_TupleVector>
            struct IsEligible
            {
                using Species = typename pmacc::math::CT::At<T_TupleVector, bmpl::int_<species>>::type;
                using Solver = typename pmacc::math::CT::At<T_TupleVector, bmpl::int_<plugin>>::type;

                using type = typename particles::traits::SpeciesEligibleForSolver<Species, Solver>::type;
            };
        };

        /* define stand alone plugins */
        using StandAlonePlugins = bmpl::vector<
            Checkpoint,
            EnergyFields
#if(ENABLE_ADIOS == 1)
            ,
            plugins::multi::Master<adios::ADIOSWriter>
#endif

#if(ENABLE_OPENPMD == 1)
            ,
            plugins::multi::Master<openPMD::openPMDWriter>
#endif

#if(PMACC_CUDA_ENABLED == 1)
            ,
            SumCurrents,
            ChargeConservation
#    if(SIMDIM == DIM3)
            ,
            IntensityPlugin
#    endif
#endif

#if(ENABLE_ISAAC == 1) && (SIMDIM == DIM3)
            ,
            isaacP::IsaacPlugin
#endif
            ,
            ResourceLog>;


        /* define field plugins */
        using UnspecializedFieldPlugins = bmpl::vector<
#if(PMACC_CUDA_ENABLED == 1)
            SliceFieldPrinterMulti<bmpl::_1>
#endif
            >;

        using AllFields = bmpl::vector<FieldB, FieldE, FieldJ>;

        using CombinedUnspecializedFieldPlugins =
            typename AllCombinations<bmpl::vector<AllFields, UnspecializedFieldPlugins>>::type;

        using FieldPlugins = typename bmpl::
            transform<CombinedUnspecializedFieldPlugins, typename TupleSpeciesPlugin::Apply<bmpl::_1>>::type;


        /* define species plugins */
        using UnspecializedSpeciesPlugins = bmpl::vector<
            plugins::multi::Master<EnergyParticles<bmpl::_1>>,
            plugins::multi::Master<CalcEmittance<bmpl::_1>>,
            plugins::multi::Master<BinEnergyParticles<bmpl::_1>>,
            CountParticles<bmpl::_1>,
            PngPlugin<Visualisation<bmpl::_1, PngCreator>>,
            plugins::transitionRadiation::TransitionRadiation<bmpl::_1>
#if(ENABLE_OPENPMD == 1)
            ,
            plugins::xrayScattering::XrayScattering<bmpl::_1>
#endif
#if(ENABLE_HDF5 == 1)
            ,
            plugins::radiation::Radiation<bmpl::_1>,
            plugins::multi::Master<ParticleCalorimeter<bmpl::_1>>
#endif
#if(ENABLE_OPENPMD == 1)
            ,
            plugins::multi::Master<PhaseSpace<particles::shapes::Counter::ChargeAssignment, bmpl::_1>>
#endif
#if(PMACC_CUDA_ENABLED == 1)
            ,
            PositionsParticles<bmpl::_1>,
            plugins::particleMerging::ParticleMerger<bmpl::_1>,
            plugins::randomizedParticleMerger::RandomizedParticleMerger<bmpl::_1>
#    if(ENABLE_HDF5 == 1)
            ,
            PerSuperCell<bmpl::_1>
#    endif
#endif
            >;

        using CombinedUnspecializedSpeciesPlugins =
            typename AllCombinations<bmpl::vector<VectorAllSpecies, UnspecializedSpeciesPlugins>>::type;

        using CombinedUnspecializedSpeciesPluginsEligible = typename bmpl::
            copy_if<CombinedUnspecializedSpeciesPlugins, typename TupleSpeciesPlugin::IsEligible<bmpl::_1>>::type;

        using SpeciesPlugins = typename bmpl::
            transform<CombinedUnspecializedSpeciesPluginsEligible, typename TupleSpeciesPlugin::Apply<bmpl::_1>>::type;

        /* create sequence with all fully specialized plugins */
        using AllPlugins = MakeSeq_t<StandAlonePlugins, FieldPlugins, SpeciesPlugins>;

        /**
         * Initializes the controller by adding all user plugins to its internal list.
         */
        virtual void init()
        {
            meta::ForEach<AllPlugins, PushBack<bmpl::_1>> pushBack;
            pushBack(plugins);
        }

    public:
        PluginController()
        {
            init();
        }

        virtual ~PluginController()
        {
        }

        void setMappingDescription(MappingDesc* cellDescription)
        {
            PMACC_ASSERT(cellDescription != nullptr);

            for(std::list<ISimulationPlugin*>::iterator iter = plugins.begin(); iter != plugins.end(); ++iter)
            {
                (*iter)->setMappingDescription(cellDescription);
            }
        }

        virtual void pluginRegisterHelp(po::options_description&)
        {
            // no help required at the moment
        }

        std::string pluginGetName() const
        {
            return "PluginController";
        }

        void notify(uint32_t)
        {
        }

        virtual void pluginUnload()
        {
            for(std::list<ISimulationPlugin*>::iterator iter = plugins.begin(); iter != plugins.end(); ++iter)
            {
                __delete(*iter);
            }
            plugins.clear();
        }
    };

} // namespace picongpu
