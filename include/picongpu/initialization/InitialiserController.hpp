/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt, Sergei Bastrakov
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/MaxwellSolver/CFLChecker.hpp"
#include "picongpu/fields/MaxwellSolver/DispersionRelationSolver.hpp"
#include "picongpu/fields/MaxwellSolver/traits/IsSubstepping.hpp"
#include "picongpu/fields/incidentField/Traits.hpp"
#include "picongpu/fields/incidentField/profiles/profiles.hpp"
#include "picongpu/initialization/IInitPlugin.hpp"
#include "picongpu/initialization/SimStartInitialiser.hpp"
#include "picongpu/particles/traits/GetDensityRatio.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/assert.hpp>
#include <pmacc/pluginSystem/PluginConnector.hpp>

namespace picongpu
{
    using namespace pmacc;


    namespace po = boost::program_options;

    class InitialiserController : public IInitPlugin
    {
    public:
        InitialiserController() = default;

        ~InitialiserController() override = default;

        /**
         * Initialize simulation state at timestep 0
         */
        void init() override
        {
            // start simulation using default values
            log<picLog::SIMULATION_STATE>("Starting simulation from timestep 0");

            SimStartInitialiser simStartInitialiser;
            Environment<>::get().DataConnector().initialise(simStartInitialiser, 0);
            eventSystem::getTransactionEvent().waitForFinished();

            log<picLog::SIMULATION_STATE>("Loading from default values finished");
        }

        /**
         * Load persistent simulation state from \p restartStep
         */
        void restart(uint32_t restartStep, const std::string restartDirectory) override
        {
            // restart simulation by loading from persistent data
            // the simulation will start after restartStep
            log<picLog::SIMULATION_STATE>("Restarting simulation from timestep %1% in directory '%2%'") % restartStep
                % restartDirectory;

            Environment<>::get().PluginConnector().restartPlugins(restartStep, restartDirectory);
            eventSystem::getTransactionEvent().waitForFinished();

            alpaka::wait(manager::Device<ComputeDevice>::get().current());

            GridController<simDim>& gc = Environment<simDim>::get().GridController();

            // avoid deadlock between not finished pmacc tasks and MPI_Barrier
            eventSystem::getTransactionEvent().waitForFinished();
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
        template<typename T_Species = boost::mpl::_1>
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
                const auto omegaP_dt = sqrt(density * charge / mass * charge / EPS0) * sim.pic.getDt();
                log<picLog::PHYSICS>("species %2%: omega_p * dt <= 0.1 ? (omega_p * dt = %1%)") % omegaP_dt
                    % FrameType::getName();
            }
        };

        /**
         * Print interesting initialization information
         */
        void printInformation() override
        {
            if(Environment<simDim>::get().GridController().getGlobalRank() == 0)
            {
                auto maxC_DT = fields::maxwellSolver::CFLChecker<fields::Solver>{}();
                auto const isSubstepping = fields::maxwellSolver::traits::IsSubstepping<fields::Solver>::value;
                auto const dtName = std::string{isSubstepping ? "substepping_dt" : "dt"};
                auto const cflMessage
                    = std::string{"Field solver condition: c * "} + dtName + " <= %1% ? (c * " + dtName + " = %2%)";
                log<picLog::PHYSICS>(cflMessage.c_str()) % maxC_DT % (SPEED_OF_LIGHT * sim.pic.getDt());

                printDispersionInformation();

                using SpeciesWithMass =
                    typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, massRatio<>>::type;
                using SpeciesWithMassCharge =
                    typename pmacc::particles::traits::FilterByFlag<SpeciesWithMass, chargeRatio<>>::type;
                meta::ForEach<SpeciesWithMassCharge, LogOmegaP<>> logOmegaP;
                log<picLog::PHYSICS>("Resolving plasma oscillations?\n"
                                     "   Estimates are based on DensityRatio to BASE_DENSITY of each species\n"
                                     "   (see: density.param, speciesDefinition.param).\n"
                                     "   It does not cover other forms of initialization");
                logOmegaP();

                const int localNrOfCells = cellDescription->getGridLayout().sizeWithoutGuardND().productOfComponents();
                log<picLog::PHYSICS>("macro particles per device: %1%")
                    % (localNrOfCells * TYPICAL_PARTICLES_PER_CELL * (pmacc::mp_size<VectorAllSpecies>::value));
                log<picLog::PHYSICS>("typical macro particle weighting: %1%")
                    % (TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE);


                log<picLog::PHYSICS>("UNIT_SPEED %1%") % UNIT_SPEED;
                log<picLog::PHYSICS>("sim.unit.time() %1%") % sim.unit.time();
                log<picLog::PHYSICS>("UNIT_LENGTH %1%") % UNIT_LENGTH;
                log<picLog::PHYSICS>("UNIT_MASS %1%") % UNIT_MASS;
                log<picLog::PHYSICS>("UNIT_CHARGE %1%") % UNIT_CHARGE;
                log<picLog::PHYSICS>("UNIT_EFIELD %1%") % UNIT_EFIELD;
                log<picLog::PHYSICS>("UNIT_BFIELD %1%") % UNIT_BFIELD;
                log<picLog::PHYSICS>("UNIT_ENERGY %1%") % UNIT_ENERGY;
            }
        }

        void notify(uint32_t) override
        {
            // nothing to do here
        }

        void pluginRegisterHelp(po::options_description& desc) override
        {
            // nothing to do here
        }

        std::string pluginGetName() const override
        {
            return "Initializers";
        }

        void setMappingDescription(MappingDesc* cellDescription) override
        {
            PMACC_ASSERT(cellDescription != nullptr);
            this->cellDescription = cellDescription;
        }

        void slide(uint32_t currentStep) override
        {
            SimStartInitialiser simStartInitialiser;
            Environment<>::get().DataConnector().initialise(simStartInitialiser, currentStep);
            eventSystem::getTransactionEvent().waitForFinished();
        }

    private:
        /*Descripe simulation area*/
        MappingDesc* cellDescription{nullptr};

        bool restartSim;
        std::string restartFile;

        /** Functor to print dispersion information for the given incident field profile
         *
         * @tparam T_Profile incident field profile
         */
        template<typename T_Profile>
        struct PrintIncidentFieldDispersion
        {
            // Calculate and print phase velocity
            void operator()() const
            {
                // Skip profiles that are not sufficiently parametrized to calculate phase velocity
                auto const printInfo = (fields::incidentField::amplitude<T_Profile> > 0.0_X);
                if(printInfo)
                {
                    auto const phaseVelocity = fields::incidentField::getPhaseVelocity<T_Profile>();
                    auto const phaseVelocityC = phaseVelocity / SPEED_OF_LIGHT;
                    auto const message = std::string{"Incident field \""} + T_Profile::getName()
                        + "\" numerical dispersion: v_phase = %1% * c";
                    log<picLog::PHYSICS>(message.c_str()) % phaseVelocityC;
                }
            }
        };

        //! Print dispersion information for the incident field lasers
        void printDispersionInformation()
        {
            using namespace fields;
            using IncidentFieldProfiles = fields::incidentField::UniqueEnabledProfiles;
            meta::ForEach<IncidentFieldProfiles, PrintIncidentFieldDispersion<boost::mpl::_1>>
                printIncidentFieldDispersion;
            printIncidentFieldDispersion();
        }
    };

} // namespace picongpu
