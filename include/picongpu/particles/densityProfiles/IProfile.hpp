/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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
                    !std::is_default_constructible_v<
                        DeferFunctor> && std::is_constructible_v<DeferFunctor, uint32_t>>* = 0)
                : T_Profile(currentStep)
            {
            }

            template<typename DeferFunctor = T_Profile>
            HINLINE IProfile(
                uint32_t currentStep,
                IdGenerator idGen,
                std::enable_if_t<
                    !std::is_default_constructible_v<
                        DeferFunctor> && std::is_constructible_v<DeferFunctor, uint32_t, IdGenerator>>* = 0)
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
