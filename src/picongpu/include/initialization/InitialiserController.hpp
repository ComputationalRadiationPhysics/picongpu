/**
 * Copyright 2013 Axel Huebl, Heiko Burau, René Widera
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
 


#ifndef INITIALISERCONTROLLER_HPP
#define	INITIALISERCONTROLLER_HPP

#include "types.h"
#include "simulation_defines.hpp"


#include "moduleSystem/ModuleConnector.hpp"

#include "fields/FieldE.hpp"
#include "fields/FieldB.hpp"

#if (ENABLE_HDF5==1)
#include "initialization/SimRestartInitialiser.hpp"
#endif
#include "initialization/SimStartInitialiser.hpp"
#include "particles/Species.hpp"

#include "initialization/IInitModule.hpp"

namespace picongpu
{
using namespace PMacc;


namespace po = boost::program_options;

class InitialiserController : public IInitModule
{
public:

    InitialiserController() :
    cellDescription(NULL),
    loadSim(false),
    restartSim(false),
    restartFile("h5"),
    xmlFile("sim.x()ml"),
    simStartInitialiser(NULL)
#if(ENABLE_SIMLIB==1)
    , simDescriptionInitialiser(NULL)
#endif
#if (ENABLE_HDF5==1)
    , simRestartInitialiser(NULL)
#endif
    {
        //ModuleConnector::getInstance().registerModule(this);
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
        if (GridController<simDim>::getInstance().getGlobalRank() == 0)
        {
            std::cout << "max weighting " << NUM_EL_PER_PARTICLE << std::endl;
            std::cout << "courant=min(deltaCellSize)/dt/c > 1.77 ? "
                << std::min(CELL_WIDTH, std::min(CELL_DEPTH, CELL_HEIGHT)) / SPEED_OF_LIGHT / DELTA_T << std::endl;
            if (gasProfile::GAS_ENABLED)
                std::cout << "omega_pe * dt <= 0.1 ? " << sqrt(GAS_DENSITY * Q_EL / M_EL * Q_EL / EPS0) * DELTA_T << std::endl;
            if (laserProfile::INIT_TIME > float_X(0.0))
                std::cout << "y-cells per wavelength: " << laserProfile::WAVE_LENGTH / CELL_HEIGHT << std::endl;
            const int localNrOfCells = cellDescription->getGridLayout().getDataSpaceWithoutGuarding().getElementCount();
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
            typedef typename boost::mpl::find<Hdf5OutputFields, FieldE>::type itFindFieldE;
            typedef typename boost::mpl::find<Hdf5OutputFields, FieldB>::type itFindFieldB;
            typedef typename boost::mpl::end< Hdf5OutputFields>::type itEnd;
            const bool restartImpossible = (boost::is_same<itFindFieldE, itEnd>::value)
                                        || (boost::is_same<itFindFieldB, itEnd>::value);
            if( restartImpossible )
                std::cout << "WARNING: HDF5 restart impossible! (dump at least "
                          << "FieldE and FieldB in hdf5Output.unitless)"
                          << std::endl;
#endif
        }

        DataSpace<simDim> globalRootCell(SubGrid<simDim>::getInstance().getSimulationBox().getGlobalOffset());

#if (ENABLE_HDF5==1)
        // restart simulation by loading from hdf5 data file
        // the simulation will start after the last saved iteration
        if (restartSim)
        {
            simRestartInitialiser = new SimRestartInitialiser<PIC_Electrons, PIC_Ions, simDim > (
                                                                                                 restartFile.c_str(),
                                                                                                 globalRootCell,
                                                                                                 cellDescription->getGridLayout().getDataSpaceWithoutGuarding());

            DataConnector::getInstance().initialise(*simRestartInitialiser, 0);

            uint32_t simulationStep = simRestartInitialiser->getSimulationStep() + 1;

            __getTransactionEvent().waitForFinished();
            delete simRestartInitialiser;
            simRestartInitialiser = NULL;

            std::cout << "Loading from hdf5 finished, can start program" << std::endl;

            return simulationStep;
        }
#endif

        // start simulation using default values
        simStartInitialiser = new SimStartInitialiser<PIC_Electrons, PIC_Ions > ();
        DataConnector::getInstance().initialise(*simStartInitialiser, 0);
        __getTransactionEvent().waitForFinished();
        delete simStartInitialiser;
        simStartInitialiser = NULL;

        std::cout << "Loading from default values finished, can start program" << std::endl;

        return 0;
    }

    void moduleLoad()
    {
        if (loadSim && restartSim)
        {
            throw ModuleException("Invalid configuration. Can't use --load and --restart together");
        }
    }

    void moduleUnload()
    {
    }

    void moduleRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ("load", po::value<bool>(&loadSim)->zero_tokens(), "load simulation from xml description")
            ("xml-infile", po::value<std::string > (&xmlFile)->default_value(xmlFile), "simulation description file")
#if (ENABLE_HDF5==1)
            ("restart", po::value<bool>(&restartSim)->zero_tokens(), "restart simulation from HDF5")
            ("restart-file", po::value<std::string > (&restartFile)->default_value(restartFile), "HDF5 file to restart simulation from")
#endif
            ;
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

        simStartInitialiser = new SimStartInitialiser<PIC_Electrons, PIC_Ions > ();
        DataConnector::getInstance().initialise(*simStartInitialiser, currentStep);
        __getTransactionEvent().waitForFinished();
        delete simStartInitialiser;
        simStartInitialiser = NULL;

    }

private:
    /*Descripe simulation area*/
    MappingDesc *cellDescription;

    // different initialisers for simulation data
    SimStartInitialiser<PIC_Electrons, PIC_Ions>* simStartInitialiser;
#if (ENABLE_HDF5==1)
    SimRestartInitialiser<PIC_Electrons, PIC_Ions, simDim> *simRestartInitialiser;
#endif
#if(ENABLE_SIMLIB==1)
    SimDescriptionInitialiser<PIC_Electrons, PIC_Ions, simDim> *simDescriptionInitialiser;
#endif

    bool loadSim;
    bool restartSim;
    std::string restartFile;
    std::string xmlFile;

};

}

#endif	/* INITIALISERCONTROLLER_HPP */

