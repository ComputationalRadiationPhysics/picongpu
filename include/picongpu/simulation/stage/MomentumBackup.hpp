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

#include "picongpu/particles/Manipulate.hpp"

#include <pmacc/particles/traits/FilterByIdentifier.hpp>

#include <cstdint>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** Functor for the stage of the PIC loop copying particles' momentums
             *  to momentumPrev1
             *
             * Only affects particle species with the momentumPrev1 attribute.
             */
            struct MomentumBackup
            {
                /** Copy the momentums
                 *
                 * @param step index of time iteration
                 */
                void operator()(uint32_t const step) const
                {
                    using pmacc::particles::traits::FilterByIdentifier;
                    using SpeciesWithMomentumPrev1 =
                        typename FilterByIdentifier<VectorAllSpecies, momentumPrev1>::type;
                    using CopyMomentum = particles::manipulators::unary::CopyAttribute<momentumPrev1, momentum>;
                    particles::manipulate<CopyMomentum, SpeciesWithMomentumPrev1>(step);
                }
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
