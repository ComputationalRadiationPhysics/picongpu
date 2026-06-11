/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/densityProfiles/IProfile.def"

#include <cstdlib>

namespace picongpu
{
    namespace densityProfiles
    {
        /** Wrapper around a given density profile functor
         *
         * Defines density profile "concept" interface and compile-time checks that
         * the given profile type is compatible to it
         *
         * @tparam T_Profile wrapped density profile functor type
         */
        template<typename T_Profile>
        struct IProfile : private T_Profile
        {
            /** create a profile functor for the given time iteration
             *
             * This constructor is only compiled if the user functor has
             * a host side constructor with one (uint32_t) or (uint32_t, IdGenerator) arguments.
             *
             * @tparam DeferFunctor is used to defer the functor type evaluation to enable/disable
             *                      the constructor
             * @param currentStep current simulation time step
             *
             * @{
             */
            template<typename DeferFunctor = T_Profile>
            HINLINE IProfile(
                uint32_t currentStep,
                IdGenerator,
                std::enable_if_t<
                    !std::is_default_constructible_v<DeferFunctor>
                    && std::is_constructible_v<DeferFunctor, uint32_t>>* = 0)
                : T_Profile(currentStep)
            {
            }

            template<typename DeferFunctor = T_Profile>
            HINLINE IProfile(
                uint32_t currentStep,
                IdGenerator idGen,
                std::enable_if_t<
                    !std::is_default_constructible_v<DeferFunctor>
                    && std::is_constructible_v<DeferFunctor, uint32_t, IdGenerator>>* = 0)
                : T_Profile(currentStep, idGen)
            {
            }

            /** @} */

            /** Calculate physical particle density value for the given cell
             *
             * It concerns real (physical, not macro-) particles.
             * The result is in units of BASE_DENSITY times PIC units of volume**-3.
             *
             * The density is assumed constant inside a cell, so the underlying
             * functor should preferably return a value in the cell center.
             *
             * @param totalCellOffset total offset from the start of the global
             *                        simulation area, including all slides [in cells]
             */
            HDINLINE float_X operator()(pmacc::DataSpace<simDim> const& totalCellOffset)
            {
                return T_Profile::operator()(totalCellOffset);
            }
        };

    } // namespace densityProfiles
} // namespace picongpu
