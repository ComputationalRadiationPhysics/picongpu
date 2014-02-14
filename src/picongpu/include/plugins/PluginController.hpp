/**
 * Copyright 2013-2014 Axel Huebl, Benjamin Schneider, Felix Schmitt, Heiko Burau, Rene Widera
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



#ifndef ANALYSISCONTROLLER_HPP
#define	ANALYSISCONTROLLER_HPP

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"


#include "particles/Species.hpp"
#include "plugins/CountParticles.hpp"
#include "plugins/EnergyParticles.hpp"
#include "plugins/EnergyFields.hpp"
#include "plugins/SumCurrents.hpp"
#include "plugins/PositionsParticles.hpp"
#include "plugins/BinEnergyParticles.hpp"
#include "plugins/LineSliceFields.hpp"

#if (ENABLE_INSITU_VOLVIS == 1)
#include "plugins/InSituVolumeRenderer.hpp"
#endif

#if(ENABLE_RADIATION == 1 && SIMDIM==DIM3)
#include "plugins/radiation/parameters.hpp"
#include "plugins/Radiation.hpp"
#endif
#include "particles/Species.hpp"
#include "simulation_classTypes.hpp"

#include "mappings/kernel/MappingDescription.hpp"

#include "plugins/LiveViewModule.hpp"
#include "plugins/IPluginModule.hpp"

#if(PIC_ENABLE_PNG==1)
#include "plugins/output/images/PngCreator.hpp"
#endif


/// That's an abstract ImageModule for Png and Binary Density output
/// \todo rename PngModule to ImageModule or similar
#include "plugins/PngModule.hpp"

#if(SIMDIM==DIM3)
#include "plugins/IntensityModule.hpp"

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

#include "plugins/IPluginModule.hpp"

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

class PluginController : public IPluginModule
{
private:

    std::list<IPluginModule*> modules;

#if (ENABLE_ELECTRONS == 1)
#if(PIC_ENABLE_PNG==1)
    typedef Visualisation<PIC_Electrons, PngCreator> ElectronsPngBuilder;
    typedef PngModule<ElectronsPngBuilder > PngImageElectrons;
#endif
    typedef ParticleDensity<PIC_Electrons, DensityToBinary, float_X> ElectronsBinaryDensityBuilder;

#if(SIMDIM==DIM3)
#if(PIC_ENABLE_PNG==1)
        typedef heiko::ParticleDensity<PIC_Electrons> HeikoParticleDensity;
        
#endif
        typedef ParticleSpectrum<PIC_Electrons> ElectronSpectrum;
        typedef SliceFieldPrinterMulti<FieldE, FIELD_E> SliceFieldEPrinter;
        typedef SliceFieldPrinterMulti<FieldB, FIELD_B> SliceFieldBPrinter;
#endif
        typedef LiveViewModule<PIC_Electrons > LiveImageElectrons;
        typedef PngModule<ElectronsBinaryDensityBuilder > BinDensityElectrons;
        typedef CountParticles<PIC_Electrons> ElectronCounter;
        typedef EnergyParticles<PIC_Electrons> EnergyElectrons;
        typedef PositionsParticles<PIC_Electrons> PositionElectrons;
        typedef BinEnergyParticles<PIC_Electrons> BinEnergyElectrons;
#if(ENABLE_RADIATION == 1 && SIMDIM==DIM3)
    typedef Radiation<PIC_Electrons> RadiationElectrons;
#endif
#endif

#if (ENABLE_IONS == 1)
#if(PIC_ENABLE_PNG==1)
    typedef Visualisation<PIC_Ions, PngCreator> IonsPngBuilder;
    typedef PngModule<IonsPngBuilder > PngImageIons;
#endif
    typedef ParticleDensity<PIC_Ions, DensityToBinary, float_X> IonsBinaryDensityBuilder;
    typedef PngModule<IonsBinaryDensityBuilder > BinDensityIons;
    typedef LiveViewModule<PIC_Ions > LiveImageIons;
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

    virtual void init()
    {
#if (ENABLE_HDF5 == 1)
        modules.push_back(new hdf5::HDF5Writer());
#endif
            
#if (ENABLE_ADIOS == 1)
        modules.push_back(new adios::ADIOSWriter());
#endif
            
        modules.push_back(new EnergyFields("EnergyFields", "energy_fields"));
        modules.push_back(new SumCurrents());
        modules.push_back(new LineSliceFields());

#if(SIMDIM==DIM3)
        modules.push_back(new FieldEnergy("FieldEnergy [keV/m^3]", "field_energy"));
#if(PIC_ENABLE_PNG==1)
        modules.push_back(new HeikoParticleDensity("HeikoParticleDensity", "heiko_pd"));
#endif
        modules.push_back(new ElectronSpectrum("Electron Spectrum", "spectrum"));
        modules.push_back(new TotalDivJ("change of total charge per timestep (single gpu)", "totalDivJ"));
        modules.push_back(new SliceFieldEPrinter("FieldE: prints a slice of the E-field", "FieldE"));
        modules.push_back(new SliceFieldBPrinter("FieldB: prints a slice of the B-field", "FieldB"));
        
        modules.push_back(new IntensityModule("Intensity", "intensity"));
#endif
        
#if (ENABLE_ELECTRONS == 1)
        modules.push_back(new LiveImageElectrons("LiveImageElectrons", "live_e"));
#if(PIC_ENABLE_PNG==1)
        modules.push_back(new PngImageElectrons("PngImageElectrons", "png_e"));
#endif
        modules.push_back(new BinDensityElectrons("BinDensityElectrons", "binDensity_e"));
        modules.push_back(new BinEnergyElectrons("BinEnergyElectrons", "bin_e"));
        modules.push_back(new ElectronCounter("ElectronsCount", "elec_cnt"));
        modules.push_back(new EnergyElectrons("EnergyElectrons", "energy_e"));
        modules.push_back(new PositionElectrons("PositionsElectrons", "pos_e"));
#endif

#if (ENABLE_IONS == 1)
        modules.push_back(new LiveImageIons("LiveImageIons", "live_i"));
#if(PIC_ENABLE_PNG==1)
        modules.push_back(new PngImageIons("PngImageIons", "png_i"));
#endif
        modules.push_back(new BinDensityIons("BinDensityIons", "binDensity_i"));
        modules.push_back(new BinEnergyIons("BinEnergyIons", "bin_i"));
        modules.push_back(new IonCounter("IonsCount", "ions_cnt"));
        modules.push_back(new EnergyIons("EnergyIons", "energy_i"));
#endif

#if(ENABLE_RADIATION == 1 && SIMDIM==DIM3)
        modules.push_back(new RadiationElectrons("RadiationElectrons", "radiation_e"));
#endif

#if (ENABLE_INSITU_VOLVIS == 1)
        modules.push_back(new InSituVolumeRenderer("InSituVolumeRenderer", "insituvolvis"));
#endif
#if (ENABLE_HDF5 == 1)
#if (ENABLE_ELECTRONS == 1)
        modules.push_back(new ElectronMakroParticleCounterPerSuperCell("ElectronsMakroParticleCounterPerSuperCell","countPerSuperCell_e"));
#endif
#if (ENABLE_IONS == 1)
        modules.push_back(new IonMakroParticleCounterPerSuperCell("IonsMakroParticleCounterPerSuperCell","countPerSuperCell_i"));
#endif
#endif
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

        for (std::list<IPluginModule*>::iterator iter = modules.begin();
             iter != modules.end();
             ++iter)
        {
            (*iter)->setMappingDescription(cellDescription);
        }
    }

    virtual void moduleRegisterHelp(po::options_description&)
    {

    }

    std::string moduleGetName() const
    {
        return "Analyser";
    }

    virtual void moduleLoad()
    {

    }

    virtual void moduleUnload()
    {
        for (std::list<IPluginModule*>::iterator iter = modules.begin();
             iter != modules.end();
             ++iter)
        {
            __delete(*iter);
        }
        modules.clear();
    }
};

}

#endif	/* ANALYSISCONTROLLER_HPP */
