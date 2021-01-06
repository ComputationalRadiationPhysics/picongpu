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

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByIdentifier.hpp>

#include <cstdint>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** Functor for the stage of the PIC loop performing FLYlite population
             *  kinetics for atomic physics
             *
             *  Only affects particle species with the populationKinetics attribute.
             */
            struct PopulationKinetics
            {
                /** Perform FLYlite population kinetics for atomic physics
                 *
                 * @param step index of time iteration
                 */
                void operator()(uint32_t const step) const
                {
                    using pmacc::particles::traits::FilterByFlag;
                    using FlyLiteIons = typename FilterByFlag<VectorAllSpecies, populationKinetics<>>::type;
                    pmacc::meta::ForEach<FlyLiteIons, particles::CallPopulationKinetics<bmpl::_1>, bmpl::_1>
                        populationKinetics;
                    populationKinetics(step);
                }
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
