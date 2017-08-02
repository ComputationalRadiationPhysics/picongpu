/* Copyright 2013-2017 Axel Huebl, Benjamin Schneider, Felix Schmitt,
 *                     Heiko Burau, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz, Erik Zenker
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
#include "picongpu/plugins/EnergyFields.hpp"
#include "picongpu/plugins/SumCurrents.hpp"
#include "picongpu/plugins/PositionsParticles.hpp"
#include "picongpu/plugins/BinEnergyParticles.hpp"
#include "picongpu/plugins/ChargeConservation.hpp"
#include "picongpu/plugins/particleMerging/ParticleMerger.hpp"
#if(ENABLE_HDF5 == 1)
#include "picongpu/plugins/radiation/parameters.hpp"
#include "picongpu/plugins/radiation/Radiation.hpp"
#include "picongpu/plugins/particleCalorimeter/ParticleCalorimeter.hpp"
#include "picongpu/plugins/PhaseSpace/PhaseSpaceMulti.hpp"
#endif

#if (ENABLE_INSITU_VOLVIS == 1)
#include "picongpu/plugins/InSituVolumeRenderer.hpp"
#endif

#include <pmacc/mappings/kernel/MappingDescription.hpp>

#include "picongpu/plugins/LiveViewPlugin.hpp"
#include "picongpu/plugins/ILightweightPlugin.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"

#include "picongpu/plugins/output/images/PngCreator.hpp"


// That's an abstract plugin for Png and Binary Density output
// \todo rename PngPlugin to ImagePlugin or similar
#include "picongpu/plugins/PngPlugin.hpp"

#if(SIMDIM==DIM3)
#include "picongpu/plugins/IntensityPlugin.hpp"
#endif
#if( PMACC_CUDA_ENABLED == 1 )
#   include "picongpu/plugins/SliceFieldPrinterMulti.hpp"
#endif
#include "picongpu/plugins/output/images/Visualisation.hpp"

#include <list>

#include "picongpu/plugins/ISimulationPlugin.hpp"

#if (ENABLE_HDF5 == 1)
#include "picongpu/plugins/hdf5/HDF5Writer.hpp"
#include "picongpu/plugins/makroParticleCounter/PerSuperCell.hpp"
#endif

#if (ENABLE_ADIOS == 1)
#include "picongpu/plugins/adios/ADIOSWriter.hpp"
#endif

#if (ENABLE_ISAAC == 1) && (SIMDIM==DIM3)
#include "picongpu/plugins/IsaacPlugin.hpp"
#endif

#include "picongpu/plugins/ResourceLog.hpp"

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

    /** apply the 1st vector component to the 2nd
     *
     * @tparam T_TupleVector vector of type pmacc::math::CT::vector<dataType,plugin>
     *                       with two components
     */
    template<typename T_TupleVector>
    struct ApplyDataToPlugin :
    bmpl::apply1<typename pmacc::math::CT::At<T_TupleVector, bmpl::int_<1> >::type,
    typename pmacc::math::CT::At<T_TupleVector, bmpl::int_<0> >::type >
    {
    };

    /* define stand alone plugins*/
    typedef bmpl::vector<
        EnergyFields,
        SumCurrents,
        ChargeConservation
#if(SIMDIM==DIM3)
      , IntensityPlugin
#endif
#if (ENABLE_INSITU_VOLVIS == 1)
      , InSituVolumeRenderer
#endif
#if (ENABLE_ADIOS == 1)
      , adios::ADIOSWriter
#endif
#if (ENABLE_HDF5 == 1)
      , hdf5::HDF5Writer
#endif
#if (ENABLE_ISAAC == 1) && (SIMDIM==DIM3)
      , isaacP::IsaacPlugin
#endif
    , ResourceLog
    > StandAlonePlugins;


    /* define field plugins */
    typedef bmpl::vector<
#if( PMACC_CUDA_ENABLED == 1 )
     SliceFieldPrinterMulti<bmpl::_1>
#endif
    > UnspecializedFieldPlugins;

    typedef bmpl::vector< FieldB, FieldE, FieldJ> AllFields;

    typedef AllCombinations<
      bmpl::vector<AllFields, UnspecializedFieldPlugins>
    >::type CombinedUnspecializedFieldPlugins;

    typedef bmpl::transform<
    CombinedUnspecializedFieldPlugins,
      ApplyDataToPlugin<bmpl::_1>
    >::type FieldPlugins;


    /* define species plugins */
    typedef bmpl::vector <
        CountParticles<bmpl::_1>,
        EnergyParticles<bmpl::_1>,
        BinEnergyParticles<bmpl::_1>,
        LiveViewPlugin<bmpl::_1>,
        PositionsParticles<bmpl::_1>,
        PngPlugin< Visualisation<bmpl::_1, PngCreator> >,
        plugins::particleMerging::ParticleMerger<bmpl::_1>
#if(ENABLE_HDF5 == 1)
      , Radiation<bmpl::_1>
      , ParticleCalorimeter<bmpl::_1>
      , PerSuperCell<bmpl::_1>
      , PhaseSpaceMulti<particles::shapes::Counter::ChargeAssignment, bmpl::_1>
#endif
    > UnspecializedSpeciesPlugins;

    typedef AllCombinations<
        bmpl::vector<VectorAllSpecies, UnspecializedSpeciesPlugins>
    >::type CombinedUnspecializedSpeciesPlugins;

    typedef bmpl::transform<
        CombinedUnspecializedSpeciesPlugins,
        ApplyDataToPlugin<bmpl::_1>
    >::type SpeciesPlugins;


    /* create sequence with all plugins*/
    typedef MakeSeq<
        StandAlonePlugins,
        FieldPlugins,
        SpeciesPlugins
    >::type AllPlugins;

    /**
     * Initialises the controller by adding all user plugins to its internal list.
     */
    virtual void init()
    {
        ForEach<AllPlugins, PushBack<bmpl::_1> > pushBack;
        pushBack(forward(plugins));
    }

public:

    PluginController()
    {
        init();
    }

    virtual ~PluginController()
    {

    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        PMACC_ASSERT(cellDescription != nullptr);

        for (std::list<ISimulationPlugin*>::iterator iter = plugins.begin();
             iter != plugins.end();
             ++iter)
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
        for (std::list<ISimulationPlugin*>::iterator iter = plugins.begin();
             iter != plugins.end();
             ++iter)
        {
            __delete(*iter);
        }
        plugins.clear();
    }
};

}
