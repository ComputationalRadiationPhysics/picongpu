/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov
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

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dataManagement/ISimulationData.hpp>

#include <boost/program_options/options_description.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** Functor for the stage of the PIC loop performing current interpolation
             *  and addition to grid values of the electromagnetic field
             */
            class CurrentInterpolationAndAdditionToEMF : public pmacc::ISimulationData
            {
            public:
                /** Register program options for current interpolation
                 *
                 * @param desc program options following boost::program_options::options_description
                 */
                void registerHelp(boost::program_options::options_description& desc);

                /** Initialize the current interpolation stage
                 *
                 * This method has to be called during initialization of the simulation.
                 * Before this method is called, the instance of CurrentInterpolation cannot be used safely.
                 */
                void init();

                /** Compute the current created by particles and add it to the current density
                 *
                 * @param step index of time iteration
                 */
                void operator()(uint32_t const step) const;

                /** Name of the solver which can be used to share this class via DataConnector */
                static std::string getName()
                {
                    return "CurrentInterpolationAndAdditionToEMF";
                }

                /**
                 * Synchronizes simulation data, meaning accessing (host side) data
                 * will return up-to-date values.
                 */
                void synchronize() override{};

                /**
                 * Return the globally unique identifier for this simulation data.
                 *
                 * @return globally unique identifier
                 */
                pmacc::SimulationDataId getUniqueId() override
                {
                    return getName();
                }

            private:
                //! Name set by program option
                std::string kindName = "none";
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
