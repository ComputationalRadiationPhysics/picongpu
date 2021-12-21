/* Copyright 2015-2022 Richard Pausch, Sergei Bastrakov
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

#include "picongpu/particles/traits/GetInterpolation.hpp"
#include "picongpu/particles/traits/GetPusher.hpp"
#include "picongpu/traits/GetMargin.hpp"


namespace picongpu
{
    namespace traits
    {
        /** Get margins of a pusher for species
         *
         * @tparam T_Species particle species type
         * @tparam T_GetLowerMargin lower margin for pusher getter type
         * @tparam T_GetUpperMargin upper margin for pusher getter type
         */
        template<typename T_Species, typename T_GetLowerMargin, typename T_GetUpperMargin>
        struct GetMarginPusher
        {
        private:
            using Interpolation = typename GetInterpolation<T_Species>::type;

        public:
            using LowerMargin =
                typename pmacc::math::CT::add<typename GetLowerMargin<Interpolation>::type, T_GetLowerMargin>::type;
            using UpperMargin =
                typename pmacc::math::CT::add<typename GetUpperMargin<Interpolation>::type, T_GetUpperMargin>::type;
        };

        /** Get lower margin of the given pusher for species
         *
         * Normally, the pusher does not have to be given explicitly.
         * However, it is needed for composite pushers
         *
         * @tparam T_Species particle species type
         * @tparam T_Pusher pusher type
         */
        template<typename T_Species, typename T_Pusher>
        struct GetLowerMarginForPusher
        {
            using type = typename traits::GetMarginPusher<
                T_Species,
                typename GetLowerMargin<T_Pusher>::type,
                typename GetUpperMargin<T_Pusher>::type>::LowerMargin;
        };

        /** Get lower margin of a pusher for species
         *
         * @tparam T_Species particle species type
         */
        template<typename T_Species>
        struct GetLowerMarginPusher : GetLowerMarginForPusher<T_Species, typename GetPusher<T_Species>::type>
        {
        };

        /** Get upper margin of the given pusher for species
         *
         * Normally, the pusher does not have to be given explicitly.
         * However, it is needed for composite pushers
         *
         * @tparam T_Species particle species type
         * @tparam T_Pusher pusher type
         */
        template<typename T_Species, typename T_Pusher>
        struct GetUpperMarginForPusher
        {
            using type = typename traits::GetMarginPusher<
                T_Species,
                typename GetLowerMargin<T_Pusher>::type,
                typename GetUpperMargin<T_Pusher>::type>::UpperMargin;
        };

        /** Get upper margin of a pusher for species
         *
         * @tparam T_Species particle species type
         */
        template<typename T_Species>
        struct GetUpperMarginPusher : GetUpperMarginForPusher<T_Species, typename GetPusher<T_Species>::type>
        {
        };

    } // namespace traits
} // namespace picongpu
