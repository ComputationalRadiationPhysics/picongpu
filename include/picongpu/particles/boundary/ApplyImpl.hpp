/* Copyright 2021-2023 Sergei Bastrakov
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

#include "picongpu/defines.hpp"
#include "picongpu/particles/boundary/Kind.hpp"

#include <cstdint>


namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            /** Functor to apply the given boundary kind to particle species
             *
             * This functor operates on the near-boundary area where the boundary conditions are active.
             * Currently, this is always a part of the GUARD area of the currently outer domains (wrt periodicity).
             * The functor would only be called for the relevant exchang types of such domains.
             * If there are particles left in this area after this functor ran, they will be removed later.
             *
             * So for standard Absorbing boundaries in GUARD the functor does not need to do anything.
             * For Periodic boundaries, there are no outer domains and so again no need to do anything here.
             *
             * Thus, the general implementation doing nothing is already suited for Absorbing and Periodic.
             * However, it probably needs to be specialized for other boundary types or other Absorbing zones.
             * In this case, the implementation must ensure particles are moved to correct supercells afterwards.
             * In the general case, by calling shiftParticles() at the end.
             *
             * @param T_kind boundary kind
             */
            template<Kind T_kind>
            struct ApplyImpl
            {
                /** Apply boundary conditions along the given outer boundary
                 *
                 * @tparam T_Species particle species type
                 *
                 * @param species particle species
                 * @param exchangeType exchange describing the active boundary @see pmacc::type::ExchangeType
                 * @param currentStep current time iteration
                 */
                template<typename T_Species>
                void operator()(T_Species& species, uint32_t exchangeType, uint32_t currentStep)
                {
                }
            };

            template<Kind T_kind, typename T_Species>
            inline void applyImpl(T_Species& species, uint32_t exchangeType, uint32_t currentStep)
            {
                ApplyImpl<T_kind>{}(species, exchangeType, currentStep);
            }

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
