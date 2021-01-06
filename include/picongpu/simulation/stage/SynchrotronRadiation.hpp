/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/particles/synchrotronPhotons/SynchrotronFunctions.hpp"

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

#include <cstdint>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** Functor for the stage of the PIC loop computing synchrotron radiation
             *
             * Only affects particle species with the synchrotronPhotons attribute.
             */
            class SynchrotronRadiation
            {
            public:
                /** Create a synchrotron radiation functor
                 *
                 * Having this in constructor is a temporary solution.
                 *
                 * @param cellDescription mapping for kernels
                 * @param functions initialized synchrotron functions
                 */
                SynchrotronRadiation(
                    MappingDesc const cellDescription,
                    particles::synchrotronPhotons::SynchrotronFunctions& functions)
                    : cellDescription(cellDescription)
                    , functions(functions)
                {
                }

                /** Ionize particles
                 *
                 * @param step index of time iteration
                 */
                void operator()(uint32_t const step) const
                {
                    using pmacc::particles::traits::FilterByFlag;
                    using SynchrotronPhotonsSpecies =
                        typename FilterByFlag<VectorAllSpecies, synchrotronPhotons<>>::type;
                    pmacc::meta::ForEach<SynchrotronPhotonsSpecies, particles::CallSynchrotronPhotons<bmpl::_1>>
                        synchrotronRadiation;
                    synchrotronRadiation(cellDescription, step, functions);
                }

            private:
                //! Mapping for kernels
                MappingDesc cellDescription;

                //! Initialized synchrotron functions
                particles::synchrotronPhotons::SynchrotronFunctions& functions;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
