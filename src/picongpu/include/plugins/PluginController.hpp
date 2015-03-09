/**
 * Copyright 2013-2014 Axel Huebl, Benjamin Schneider, Felix Schmitt, 
 *                     Heiko Burau, Rene Widera, Richard Pausch
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

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include "plugins/CountParticles.hpp"
#include "plugins/EnergyParticles.hpp"
#include "plugins/EnergyFields.hpp"
#include "plugins/SumCurrents.hpp"
#include "plugins/PositionsParticles.hpp"
#include "plugins/BinEnergyParticles.hpp"
#if(ENABLE_HDF5 == 1)
#include "plugins/PhaseSpace/PhaseSpaceMulti.hpp"
#endif

#if (ENABLE_INSITU_VOLVIS == 1)
#include "plugins/InSituVolumeRenderer.hpp"
#endif

#if(ENABLE_RADIATION == 1)
#include "plugins/radiation/parameters.hpp"
#include "plugins/radiation/Radiation.hpp"
#endif

#include "simulation_classTypes.hpp"

#include "mappings/kernel/MappingDescription.hpp"

#include "plugins/LiveViewPlugin.hpp"
#include "plugins/ILightweightPlugin.hpp"
#include "plugins/ISimulationPlugin.hpp"

#if(PIC_ENABLE_PNG==1)
#include "plugins/output/images/PngCreator.hpp"
#endif


/// That's an abstract plugin for Png and Binary Density output
/// \todo rename PngPlugin to ImagePlugin or similar
#include "plugins/PngPlugin.hpp"

#if(SIMDIM==DIM3)
#include "plugins/IntensityPlugin.hpp"

#include "plugins/FieldEnergy.hpp"
#if(PIC_ENABLE_PNG==1)
#include "plugins/ParticleDensity.hpp"
#endif
#include "plugins/ParticleSpectrum.hpp"
#include "plugins/TotalDivJ.hpp"
#include "plugins/SliceFieldPrinterMulti.hpp"
#endif

#include "plugins/output/images/Visualisation.hpp"

#include "plugins/output/images/DensityToBinary.hpp"
#include "plugins/output/images/ParticleDensity.hpp"

#include <list>

#include "plugins/ISimulationPlugin.hpp"

#if (ENABLE_HDF5 == 1)
#include "plugins/hdf5/HDF5Writer.hpp"
#include "plugins/makroParticleCounter/PerSuperCell.hpp"
#endif

#if (ENABLE_ADIOS == 1)
#include "plugins/adios/ADIOSWriter.hpp"
#endif

namespace picongpu
{

using namespace PMacc;

/**
 * Plugin management controller for user-level plugins.
 */
class PluginController : public ILightweightPlugin
{
private:

    std::list<ISimulationPlugin*> plugins;

#if (ENABLE_ELECTRONS == 1)
#if(PIC_ENABLE_PNG==1)
    typedef Visualisation<PIC_Electrons, PngCreator> ElectronsPngBuilder;
    typedef PngPlugin<ElectronsPngBuilder > PngImageElectrons;
#endif
    typedef ParticleDensity<PIC_Electrons, DensityToBinary, float_X> ElectronsBinaryDensityBuilder;

#if(ENABLE_HDF5 == 1)
    /* speciesParticleShape::ParticleShape::ChargeAssignment */
    typedef PhaseSpaceMulti<particles::shapes::Counter::ChargeAssignment, PIC_Electrons> PhaseSpaceElectrons;
#endif
#if(SIMDIM==DIM3)
#if(PIC_ENABLE_PNG==1)
    typedef heiko::ParticleDensity<PIC_Electrons> HeikoParticleDensity;
#endif

    typedef ParticleSpectrum<PIC_Electrons> ElectronSpectrum;
    typedef SliceFieldPrinterMulti<FieldE> SliceFieldEPrinter;
    typedef SliceFieldPrinterMulti<FieldB> SliceFieldBPrinter;
    typedef SliceFieldPrinterMulti<FieldJ> SliceFieldJPrinter;
#endif

    typedef LiveViewPlugin<PIC_Electrons > LiveImageElectrons;
    typedef PngPlugin<ElectronsBinaryDensityBuilder > BinDensityElectrons;
    typedef CountParticles<PIC_Electrons> ElectronCounter;
    typedef EnergyParticles<PIC_Electrons> EnergyElectrons;
    typedef PositionsParticles<PIC_Electrons> PositionElectrons;
    typedef BinEnergyParticles<PIC_Electrons> BinEnergyElectrons;
#if(ENABLE_RADIATION == 1)
    typedef Radiation<PIC_Electrons> RadiationElectrons;
#endif
#endif

#if (ENABLE_IONS == 1)
#if(PIC_ENABLE_PNG==1)
    typedef Visualisation<PIC_Ions, PngCreator> IonsPngBuilder;
    typedef PngPlugin<IonsPngBuilder > PngImageIons;
#endif
#if(ENABLE_HDF5 == 1)
    /* speciesParticleShape::ParticleShape::ChargeAssignment */
    typedef PhaseSpaceMulti<particles::shapes::Counter::ChargeAssignment, PIC_Ions> PhaseSpaceIons;
#endif
    typedef ParticleDensity<PIC_Ions, DensityToBinary, float_X> IonsBinaryDensityBuilder;
    typedef PngPlugin<IonsBinaryDensityBuilder > BinDensityIons;
    typedef LiveViewPlugin<PIC_Ions > LiveImageIons;
    typedef CountParticles<PIC_Ions> IonCounter;
    typedef EnergyParticles<PIC_Ions> EnergyIons;
    typedef BinEnergyParticles<PIC_Ions> BinEnergyIons;
#endif

#if (ENABLE_HDF5 == 1)
#if (ENABLE_ELECTRONS == 1)
    typedef PerSuperCell<PIC_Electrons> ElectronMakroParticleCounterPerSuperCell;
#endif
#if (ENABLE_IONS == 1)
    typedef PerSuperCell<PIC_Ions> IonMakroParticleCounterPerSuperCell;
#endif
#endif

    /**
     * Initialises the controller by adding all user plugins to its internal list.
     */
    virtual void init()
    {
#if (ENABLE_HDF5 == 1)
        plugins.push_back(new hdf5::HDF5Writer());
#endif

#if (ENABLE_ADIOS == 1)
        plugins.push_back(new adios::ADIOSWriter());
#endif

        plugins.push_back(new EnergyFields("EnergyFields", "energy_fields"));
        plugins.push_back(new SumCurrents());

#if(SIMDIM==DIM3)
        plugins.push_back(new FieldEnergy("FieldEnergy [keV/m^3]", "field_energy"));
#if(PIC_ENABLE_PNG==1)
        plugins.push_back(new HeikoParticleDensity("HeikoParticleDensity", "heiko_pd"));
#endif
        plugins.push_back(new ElectronSpectrum("Electron Spectrum", "spectrum"));
        plugins.push_back(new TotalDivJ("change of total charge per timestep (single gpu)", "totalDivJ"));
        plugins.push_back(new SliceFieldEPrinter("FieldE: prints a slice of the E-field", "FieldE"));
        plugins.push_back(new SliceFieldBPrinter("FieldB: prints a slice of the B-field", "FieldB"));
        plugins.push_back(new SliceFieldJPrinter("FieldJ: prints a slice of the current-field", "FieldJ"));

        plugins.push_back(new IntensityPlugin("Intensity", "intensity"));
#endif

#if (ENABLE_ELECTRONS == 1)
#if(ENABLE_HDF5 == 1)
        plugins.push_back(new PhaseSpaceElectrons("PhaseSpace Electrons", "ps_e"));
#endif
        plugins.push_back(new LiveImageElectrons("LiveImageElectrons", "live_e"));
#if(PIC_ENABLE_PNG==1)
        plugins.push_back(new PngImageElectrons("PngImageElectrons", "png_e"));
#endif
        plugins.push_back(new BinDensityElectrons("BinDensityElectrons", "binDensity_e"));
        plugins.push_back(new BinEnergyElectrons("BinEnergyElectrons", "bin_e"));
        plugins.push_back(new ElectronCounter("ElectronsCount", "elec_cnt"));
        plugins.push_back(new EnergyElectrons("EnergyElectrons", "energy_e"));
        plugins.push_back(new PositionElectrons("PositionsElectrons", "pos_e"));
#endif

#if (ENABLE_IONS == 1)
#if(ENABLE_HDF5 == 1)
        plugins.push_back(new PhaseSpaceIons("PhaseSpace Ions", "ps_i"));
#endif
        plugins.push_back(new LiveImageIons("LiveImageIons", "live_i"));
#if(PIC_ENABLE_PNG==1)
        plugins.push_back(new PngImageIons("PngImageIons", "png_i"));
#endif
        plugins.push_back(new BinDensityIons("BinDensityIons", "binDensity_i"));
        plugins.push_back(new BinEnergyIons("BinEnergyIons", "bin_i"));
        plugins.push_back(new IonCounter("IonsCount", "ions_cnt"));
        plugins.push_back(new EnergyIons("EnergyIons", "energy_i"));
#endif

#if(ENABLE_RADIATION == 1)
        plugins.push_back(new RadiationElectrons("RadiationElectrons", "radiation_e"));
#endif

#if (ENABLE_INSITU_VOLVIS == 1)
        plugins.push_back(new InSituVolumeRenderer("InSituVolumeRenderer", "insituvolvis"));
#endif
#if (ENABLE_HDF5 == 1)
#if (ENABLE_ELECTRONS == 1)
        plugins.push_back(new ElectronMakroParticleCounterPerSuperCell("ElectronsMakroParticleCounterPerSuperCell","countPerSuperCell_e"));
#endif
#if (ENABLE_IONS == 1)
        plugins.push_back(new IonMakroParticleCounterPerSuperCell("IonsMakroParticleCounterPerSuperCell","countPerSuperCell_i"));
#endif
#endif

        /**
         * Add your plugin here, guard with pragmas if it depends on compile-time switches.
         * Plugins must be heap-allocated (use 'new').
         * Plugins are free'd automatically.
         * Plugins should use a short but descriptive prefix for all command line parameters, e.g.
         * 'my_plugin.period', or 'my_plugin.parameter'.
         */
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
        assert(cellDescription != NULL);

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
