/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
