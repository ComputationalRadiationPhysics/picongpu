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

#include "picongpu/simulation_defines.hpp"

#include <boost/program_options.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            namespace detail
            {
                struct IApplyFieldBackground
                {
                    virtual void add(uint32_t const step) = 0;
                    virtual void subtract(uint32_t const step) = 0;

                    virtual ~IApplyFieldBackground() = default;
                };

            } // namespace detail
            //! Functor for the stage of the PIC loop applying field background
            class FieldBackground
            {
            public:
                /** Register program options for field background
                 *
                 * @param desc program options following boost::program_options::options_description
                 */
                void registerHelp(po::options_description& desc)
                {
                    desc.add_options()(
                        "fieldBackground.duplicateFields",
                        po::value<bool>(&duplicateFields)->zero_tokens(),
                        "duplicate E and B field storage inside field background to improve its performance "
                        "and potentially avoid some numerical noise at cost of using more memory, "
                        "only affects the fields with activated background");
                }

                /** Initialize field background stage
                 *
                 * This method must be called once before calling add(), subtract() and fillSimulation().
                 * The initialization has to be delayed for this class as it needs registerHelp() like the plugins do.
                 *
                 * @param cellDescription mapping for kernels
                 */
                void init(MappingDesc const cellDescription);

                /** Add field background to the electromagnetic field
                 *
                 * Affects data sets named FieldE::getName(), FieldB::getName().
                 * As the result of this operation, they will have a sum of old values and background values.
                 *
                 * @param step index of time iteration
                 */
                void add(uint32_t const step);

                /** Subtract field background from the electromagnetic field
                 *
                 * Affects data sets named FieldE::getName(), FieldB::getName().
                 * As the result of this operation, they will have values like before the last call to add().
                 *
                 * Warning: when fieldBackground.duplicateFields is enabled, the fields are assumed to not have changed
                 * since the call to add(). Having fieldBackground.duplicateFields disabled does not rely on this.
                 * However, this assumption should generally hold true in the PIC computational loop.
                 *
                 * @param step index of time iteration
                 */
                void subtract(uint32_t const step);

            private:
                //! Check if this class was properly initialized, throws when failed
                void checkInitialization() const
                {
                    if(!applyE || !applyB)
                        throw std::runtime_error("simulation::stage::FieldBackground used without init() called");
                }

                //! Object to apply background to field E
                std::unique_ptr<detail::IApplyFieldBackground> applyE;

                //! Object to apply background to field B
                std::unique_ptr<detail::IApplyFieldBackground> applyB;

                //! Flag to store duplicates fields with enabled backgrounds
                bool duplicateFields = false;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
