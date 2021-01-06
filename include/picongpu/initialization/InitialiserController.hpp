/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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

#include <pmacc/Environment.hpp>

#include <pmacc/pluginSystem/PluginConnector.hpp>

#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/laserProfiles/profiles.hpp"
#include "picongpu/particles/traits/GetDensityRatio.hpp"

#include "picongpu/initialization/SimStartInitialiser.hpp"

#include "picongpu/initialization/IInitPlugin.hpp"
#include <pmacc/assert.hpp>

#include <boost/mpl/find.hpp>

namespace picongpu
{
    using namespace pmacc;


    namespace po = boost::program_options;

    class InitialiserController : public IInitPlugin
    {
    public:
        InitialiserController() : cellDescription(nullptr)
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
            log<picLog::SIMULATION_STATE>("Starting simulation from timestep 0");

            SimStartInitialiser simStartInitialiser;
            Environment<>::get().DataConnector().initialise(simStartInitialiser, 0);
            __getTransactionEvent().waitForFinished();

            log<picLog::SIMULATION_STATE>("Loading from default values finished");
        }

        /**
         * Load persistent simulation state from \p restartStep
         */
        virtual void restart(uint32_t restartStep, const std::string restartDirectory)
        {
            // restart simulation by loading from persistent data
            // the simulation will start after restartStep
            log<picLog::SIMULATION_STATE>("Restarting simulation from timestep %1% in directory '%2%'") % restartStep
                % restartDirectory;

            Environment<>::get().PluginConnector().restartPlugins(restartStep, restartDirectory);
            __getTransactionEvent().waitForFinished();

            CUDA_CHECK(cuplaDeviceSynchronize());
            CUDA_CHECK(cuplaGetLastError());

            GridController<simDim>& gc = Environment<simDim>::get().GridController();

            // avoid deadlock between not finished pmacc tasks and MPI_Barrier
            __getTransactionEvent().waitForFinished();
            /* can be spared for better scalings, but guarantees the user
             * that the restart was successful */
            MPI_CHECK(MPI_Barrier(gc.getCommunicator().getMPIComm()));

            log<picLog::SIMULATION_STATE>("Loading from persistent data finished");
        }

        /** Log omega_p for each species
         *
         * Calculate omega_p for each given species and create a `picLog::PHYSICS`
         * log message
         */
        template<typename T_Species = bmpl::_1>
        struct LogOmegaP
        {
            void operator()()
            {
                /* The omega_p calculation is based on species' densityRatio
                 * relative to the BASE_DENSITY. Thus, it is only accurate
                 * for species with macroparticles sampled by density,
                 * but not necessarily for derived ones.
                 */
                using FrameType = typename T_Species::FrameType;
                const float_32 charge = frame::getCharge<FrameType>();
                const float_32 mass = frame::getMass<FrameType>();
                const auto densityRatio = traits::GetDensityRatio<T_Species>::type::getValue();
                const auto density = BASE_DENSITY * densityRatio;
                log<picLog::PHYSICS>("species %2%: omega_p * dt <= 0.1 ? %1%")
                    % (sqrt(density * charge / mass * charge / EPS0) * DELTA_T) % FrameType::getName();
            }
        };

        /**
         * Print interesting initialization information
         */
        virtual void printInformation()
        {
            if(Environment<simDim>::get().GridController().getGlobalRank() == 0)
            {
                log<picLog::PHYSICS>("Courant c*dt <= %1% ? %2%") % (1. / math::sqrt(INV_CELL2_SUM))
                    % (SPEED_OF_LIGHT * DELTA_T);

                using SpeciesWithMass =
                    typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, massRatio<>>::type;
                using SpeciesWithMassCharge =
                    typename pmacc::particles::traits::FilterByFlag<SpeciesWithMass, chargeRatio<>>::type;
                meta::ForEach<SpeciesWithMassCharge, LogOmegaP<>> logOmegaP;
                log<picLog::PHYSICS>("Resolving plasma oscillations?\n"
                                     "   Estimates are based on DensityRatio to BASE_DENSITY of each species\n"
                                     "   (see: density.param, speciesDefinition.param).\n"
                                     "   It and does not cover other forms of initialization");
                logOmegaP();

                if(fields::laserProfiles::Selected::INIT_TIME > float_X(0.0))
                    log<picLog::PHYSICS>("y-cells per wavelength: %1%")
                        % (fields::laserProfiles::Selected::WAVE_LENGTH / CELL_HEIGHT);
                const int localNrOfCells
                    = cellDescription->getGridLayout().getDataSpaceWithoutGuarding().productOfComponents();
                log<picLog::PHYSICS>("macro particles per device: %1%")
                    % (localNrOfCells * particles::TYPICAL_PARTICLES_PER_CELL
                       * (bmpl::size<VectorAllSpecies>::type::value));
                log<picLog::PHYSICS>("typical macro particle weighting: %1%")
                    % (particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE);


                log<picLog::PHYSICS>("UNIT_SPEED %1%") % UNIT_SPEED;
                log<picLog::PHYSICS>("UNIT_TIME %1%") % UNIT_TIME;
                log<picLog::PHYSICS>("UNIT_LENGTH %1%") % UNIT_LENGTH;
                log<picLog::PHYSICS>("UNIT_MASS %1%") % UNIT_MASS;
                log<picLog::PHYSICS>("UNIT_CHARGE %1%") % UNIT_CHARGE;
                log<picLog::PHYSICS>("UNIT_EFIELD %1%") % UNIT_EFIELD;
                log<picLog::PHYSICS>("UNIT_BFIELD %1%") % UNIT_BFIELD;
                log<picLog::PHYSICS>("UNIT_ENERGY %1%") % UNIT_ENERGY;
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

        virtual void setMappingDescription(MappingDesc* cellDescription)
        {
            PMACC_ASSERT(cellDescription != nullptr);
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
        MappingDesc* cellDescription;

        bool restartSim;
        std::string restartFile;
    };

} // namespace picongpu
