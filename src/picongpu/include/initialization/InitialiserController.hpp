/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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

#include "Environment.hpp"

#include "moduleSystem/ModuleConnector.hpp"

#include "fields/FieldE.hpp"
#include "fields/FieldB.hpp"


#if (ENABLE_HDF5==1)
#include "initialization/SimRestartInitialiser.hpp"
#endif

#include "initialization/SimStartInitialiser.hpp"
#include "particles/Species.hpp"

#include "initialization/IInitModule.hpp"

#include <boost/mpl/find.hpp>

namespace picongpu
{
using namespace PMacc;


namespace po = boost::program_options;

class InitialiserController : public IInitModule
{
public:

    InitialiserController() :
    cellDescription(NULL),
    restartSim(false),
    restartFile("h5")
    {
        //Environment<>::get().ModuleConnector().registerModule(this);
    }

    virtual ~InitialiserController()
    {
    }

    /**
     * Initializes simulation state.
     * 
     * @return returns the first step of the simulation
     */
    virtual uint32_t init()
    {
        if (Environment<simDim>::get().GridController().getGlobalRank() == 0)
        {
            std::cout << "max weighting " << NUM_EL_PER_PARTICLE << std::endl;
            
            float_X shortestSide=cellSize.x();
            for(uint32_t d=1;d<simDim;++d)
                shortestSide=std::min(shortestSide,cellSize[d]);
                        
            std::cout << "courant=min(deltaCellSize)/dt/c > 1.77 ? "<< 
                         shortestSide / SPEED_OF_LIGHT / DELTA_T << std::endl;

            if (gasProfile::GAS_ENABLED)
                std::cout << "omega_pe * dt <= 0.1 ? " << sqrt(GAS_DENSITY * Q_EL / M_EL * Q_EL / EPS0) * DELTA_T << std::endl;
            if (laserProfile::INIT_TIME > float_X(0.0))
                std::cout << "y-cells per wavelength: " << laserProfile::WAVE_LENGTH / CELL_HEIGHT << std::endl;
            const int localNrOfCells = cellDescription->getGridLayout().getDataSpaceWithoutGuarding().productOfComponents();
            std::cout << "macro particles per gpu: "
                << localNrOfCells * particleInit::NUM_PARTICLES_PER_CELL * (1 + 1 * ENABLE_IONS) << std::endl;
            std::cout << "typical macro particle weighting: " << NUM_EL_PER_PARTICLE << std::endl;

            //const float_X y_R = M_PI * laserProfile::W0 * laserProfile::W0 / laserProfile::WAVE_LENGTH; //rayleigh length (in y-direction)
            //std::cout << "focus/y_Rayleigh: " << laserProfile::FOCUS_POS / y_R << std::endl;

            std::cout << "UNIT_SPEED" << " " << UNIT_SPEED << std::endl;
            std::cout << "UNIT_TIME" << " " << UNIT_TIME << std::endl;
            std::cout << "UNIT_LENGTH" << " " << UNIT_LENGTH << std::endl;
            std::cout << "UNIT_MASS" << " " << UNIT_MASS << std::endl;
            std::cout << "UNIT_CHARGE" << " " << UNIT_CHARGE << std::endl;
            std::cout << "UNIT_EFIELD" << " " << UNIT_EFIELD << std::endl;
            std::cout << "UNIT_BFIELD" << " " << UNIT_BFIELD << std::endl;
            std::cout << "UNIT_ENERGY" << " " << UNIT_ENERGY << std::endl;
     
#if (ENABLE_HDF5==1)
            // check for HDF5 restart capability
            typedef typename boost::mpl::find<FileOutputFields, FieldE>::type itFindFieldE;
            typedef typename boost::mpl::find<FileOutputFields, FieldB>::type itFindFieldB;
            typedef typename boost::mpl::end< FileOutputFields>::type itEnd;
            const bool restartImpossible = (boost::is_same<itFindFieldE, itEnd>::value)
                                        || (boost::is_same<itFindFieldB, itEnd>::value);
            if( restartImpossible )
                std::cout << "WARNING: HDF5 restart impossible! (dump at least "
                          << "FieldE and FieldB in hdf5Output.unitless)"
                          << std::endl;
#endif
        }

#if (ENABLE_HDF5==1)
        // restart simulation by loading from hdf5 data file
        // the simulation will start after the last saved iteration
        if (restartSim)
        {
            SimRestartInitialiser<PIC_Electrons, PIC_Ions, simDim> simRestartInitialiser(
                restartFile.c_str(), cellDescription->getGridLayout().getDataSpaceWithoutGuarding());

            Environment<>::get().DataConnector().initialise(simRestartInitialiser, 0);

            uint32_t simulationStep = simRestartInitialiser.getSimulationStep() + 1;

            __getTransactionEvent().waitForFinished();

            log<picLog::SIMULATION_STATE > ("Loading from hdf5 finished, can start program");

            return simulationStep;
        } else
#endif
        {
            // start simulation using default values
            SimStartInitialiser<PIC_Electrons, PIC_Ions> simStartInitialiser;
            Environment<>::get().DataConnector().initialise(simStartInitialiser, 0);
            __getTransactionEvent().waitForFinished();

            log<picLog::SIMULATION_STATE > ("Loading from default values finished, can start program");
        }

        return 0;
    }

    void moduleLoad()
    {
    }

    void moduleUnload()
    {
    }

    void moduleRegisterHelp(po::options_description& desc)
    {
#if (ENABLE_HDF5==1)
        desc.add_options()

            ("restart", po::value<bool>(&restartSim)->zero_tokens(), "restart simulation from HDF5")
            ("restart-file", po::value<std::string > (&restartFile)->default_value(restartFile), "HDF5 file to restart simulation from")
            ;
#endif
    }

    std::string moduleGetName() const
    {
        return "Initializers";
    }

    virtual void setMappingDescription(MappingDesc *cellDescription)
    {
        assert(cellDescription != NULL);
        this->cellDescription = cellDescription;
    }

    virtual void slide(uint32_t currentStep)
    {
        SimStartInitialiser<PIC_Electrons, PIC_Ions> simStartInitialiser;
        Environment<>::get().DataConnector().initialise(simStartInitialiser, currentStep);
        __getTransactionEvent().waitForFinished();
    }

private:
    /*Descripe simulation area*/
    MappingDesc *cellDescription;

    bool restartSim;
    std::string restartFile;

};

} //namespace picongpu
