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

#include "pluginSystem/PluginConnector.hpp"

#include "fields/FieldE.hpp"
#include "fields/FieldB.hpp"

#include "initialization/SimStartInitialiser.hpp"

#include "initialization/IInitPlugin.hpp"

#include <boost/mpl/find.hpp>

namespace picongpu
{
using namespace PMacc;


namespace po = boost::program_options;

class InitialiserController : public IInitPlugin
{
public:

    InitialiserController() :
    cellDescription(NULL)
    {
    }

    virtual ~InitialiserController()
    {
    }

    /**
     * Initialize simulation state at timestep 0
     */
    virtual void init()
    {
        // start simulation using default values
        log<picLog::SIMULATION_STATE > ("Starting simulation from timestep 0");

        SimStartInitialiser simStartInitialiser;
        Environment<>::get().DataConnector().initialise(simStartInitialiser, 0);
        __getTransactionEvent().waitForFinished();

        log<picLog::SIMULATION_STATE > ("Loading from default values finished");
    }

    /**
     * Load persistent simulation state from \p restartStep
     */
    virtual void restart(uint32_t restartStep, const std::string restartDirectory)
    {
        // restart simulation by loading from persistent data
        // the simulation will start after restartStep
        log<picLog::SIMULATION_STATE > ("Restarting simulation from timestep %1% in directory '%2%'") %
            restartStep % restartDirectory;

        Environment<>::get().PluginConnector().restartPlugins(restartStep, restartDirectory);
        __getTransactionEvent().waitForFinished();

        log<picLog::SIMULATION_STATE > ("Loading from persistent data finished");
    }

    /**
     * Print interesting initialization information
     */
    virtual void printInformation()
    {
        if (Environment<simDim>::get().GridController().getGlobalRank() == 0)
        {
            log<picLog::PHYSICS >("Courant c*dt <= %1% ? %2%") %
                                 (1./math::sqrt(INV_CELL2_SUM)) %
                                 (SPEED_OF_LIGHT * DELTA_T);

//            if (gasProfile::GAS_ENABLED)
//                log<picLog::PHYSICS >("omega_pe * dt <= 0.1 ? %1%") %
//                                     (sqrt(GAS_DENSITY * Q_EL / M_EL * Q_EL / EPS0) * DELTA_T);
            if (laserProfile::INIT_TIME > float_X(0.0))
                log<picLog::PHYSICS >("y-cells per wavelength: %1%") %
                                     (laserProfile::WAVE_LENGTH / CELL_HEIGHT);
            const int localNrOfCells = cellDescription->getGridLayout().getDataSpaceWithoutGuarding().productOfComponents();
            log<picLog::PHYSICS >("macro particles per gpu: %1%") %
                                 (localNrOfCells * particles::TYPICAL_PARTICLES_PER_CELL * (1 + 1 * ENABLE_IONS));
            log<picLog::PHYSICS >("typical macro particle weighting: %1%") % (particles::TYPICAL_NUM_PARTICLE_PER_MAKROPARTICLE);

            //const float_X y_R = M_PI * laserProfile::W0 * laserProfile::W0 / laserProfile::WAVE_LENGTH; //rayleigh length (in y-direction)
            //std::cout << "focus/y_Rayleigh: " << laserProfile::FOCUS_POS / y_R << std::endl;

            log<picLog::PHYSICS >("UNIT_SPEED %1%") % UNIT_SPEED;
            log<picLog::PHYSICS >("UNIT_TIME %1%") % UNIT_TIME;
            log<picLog::PHYSICS >("UNIT_LENGTH %1%") % UNIT_LENGTH;
            log<picLog::PHYSICS >("UNIT_MASS %1%") % UNIT_MASS;
            log<picLog::PHYSICS >("UNIT_CHARGE %1%") % UNIT_CHARGE;
            log<picLog::PHYSICS >("UNIT_EFIELD %1%") % UNIT_EFIELD;
            log<picLog::PHYSICS >("UNIT_BFIELD %1%") % UNIT_BFIELD;
            log<picLog::PHYSICS >("UNIT_ENERGY %1%") % UNIT_ENERGY;
        }
    }

    void notify(uint32_t)
    {
        // nothing to do here
    }

    void pluginRegisterHelp(po::options_description& desc)
    {
        // nothing to do here
    }

    std::string pluginGetName() const
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
        SimStartInitialiser simStartInitialiser;
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
