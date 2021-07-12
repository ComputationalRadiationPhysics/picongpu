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

#include "picongpu/simulation_defines.hpp"

#include <cstdint>
#include <memory>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** Functor for the stage of the PIC loop computing Bremsstrahlung
             *
             * Only affects particle species with the bremsstrahlungIons attribute.
             */
            class Bremsstrahlung
            {
            public:
                Bremsstrahlung();

                //! Copy construction is forbidden
                Bremsstrahlung(Bremsstrahlung const&) = delete;

                //! Destroy Bremsstrahlung stage
                ~Bremsstrahlung();

                /** Initialize Bremsstrahlung stage
                 *
                 * This method must be called once before calling operator().
                 *
                 * @param cellDescription mapping for kernels
                 */
                void init(MappingDesc const cellDescription);

                /** Ionize particles
                 *
                 * @param step index of time iteration
                 */
                void operator()(uint32_t const step) const;

            private:
                //! Implementation
                class Impl;

                //! Pointer to implementation
                std::unique_ptr<const Impl> pImpl;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
