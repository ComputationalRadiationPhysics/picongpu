/* Copyright 2024 Rene Widera
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

#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"

#include <pmacc/meta/AllCombinations.hpp>

#include <list>
#include <memory>


namespace picongpu
{
    using namespace pmacc;

    /** plugin registry entry interface */
    struct IRegistryEntry
    {
        virtual std::shared_ptr<ISimulationPlugin> createPlugin() = 0;
    };

    /** factory used to create a plugin at runtime */
    template<typename T_Plugin>
    struct RegistryEntry : IRegistryEntry
    {
        std::shared_ptr<ISimulationPlugin> createPlugin() override
        {
            return std::make_shared<T_Plugin>();
        }
    };

    /** singleton to register plugins which get created by PluginController during the startup */
    struct PluginRegistry
    {
        /** combines a plugins with all species
         *
         * Trait SpeciesEligibleForSolver is evaluated to check if a plugin support using a species.
         *
         * @tparam T_Plugin
         */
        template<typename T_Plugin>
        static void registerSpeciesPlugin()
        {
            using CombinedUnspecializedSpeciesPlugins = pmacc::AllCombinations<VectorAllSpecies, MakeSeq_t<T_Plugin>>;

            using CombinedUnspecializedSpeciesPluginsEligible
                = pmacc::mp_copy_if<CombinedUnspecializedSpeciesPlugins, TupleSpeciesPlugin::IsEligible>;

            using SpeciesPlugins
                = pmacc::mp_transform<TupleSpeciesPlugin::Apply, CombinedUnspecializedSpeciesPluginsEligible>;


            meta::ForEach<SpeciesPlugins, PushBack<boost::mpl::_1>> pushBack;
            pushBack(PluginRegistry::get().pluginList);
        }

        template<typename T_Plugin>
        static void registerStandAlonePlugin()
        {
            meta::ForEach<MakeSeq_t<T_Plugin>, PushBack<boost::mpl::_1>> pushBack;
            pushBack(PluginRegistry::get().pluginList);
        }

        static PluginRegistry& get()
        {
            static PluginRegistry instance = PluginRegistry{};
            return instance;
        }

        /** get a list of factories to create plugins
         *
         * @return list of factories where createPlugin() can be called on each entry
         */
        auto getPluginFactories() const
        {
            return pluginList;
        }

    private:
        std::list<std::shared_ptr<IRegistryEntry>> pluginList;

        PluginRegistry() = default;

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

        template<typename T_Type>
        struct PushBack
        {
            template<typename T>
            void operator()(T& list)
            {
                list.push_back(std::make_shared<RegistryEntry<T_Type>>());
            }
        };
    };

} // namespace picongpu

/** Create a static objects which registers a plugin automatically at the start of the application to the registry.
 *
 * @param counter compile time unique counter __COUNTER__
 * @param registerOperation method which is called on the singleton PluginRegistry to register the plugin
 *                          can be registerStandAlonePlugin or registerSpeciesPlugin
 * @param ... plugin type signature
 */
#define PIC_REGISTER_PLUGIN_DO(counter, registerOperation, ...)                                                       \
                                                                                                                      \
    namespace picongpu::plugin                                                                                        \
    {                                                                                                                 \
        struct plugin_##counter                                                                                       \
        {                                                                                                             \
            plugin_##counter()                                                                                        \
            {                                                                                                         \
                picongpu::PluginRegistry::registerOperation<__VA_ARGS__>();                                           \
            }                                                                                                         \
        };                                                                                                            \
    }                                                                                                                 \
    static picongpu::plugin::plugin_##counter plugin_instance##counter                                                \
    {                                                                                                                 \
    }

/** Register a species independent plugin class to PIConGPU
 *
 * @param ... plugin type signature
 */
#define PIC_REGISTER_PLUGIN(...) PIC_REGISTER_PLUGIN_DO(__COUNTER__, registerStandAlonePlugin, __VA_ARGS__)

/** Register a species dependent plugin class to PIConGPU
 *
 * The plugin is combined with all possible species in case the checked with the trait SpeciesEligibleForSolver is
 * evaluated to true.
 *
 * @param ... plugin type signature
 */
#define PIC_REGISTER_SPECIES_PLUGIN(...) PIC_REGISTER_PLUGIN_DO(__COUNTER__, registerSpeciesPlugin, __VA_ARGS__)
