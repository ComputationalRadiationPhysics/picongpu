/* Copyright 2015-2021 Richard Pausch, Sergei Bastrakov
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
#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/particles/traits/GetInterpolation.hpp"
#include "picongpu/particles/traits/GetPusher.hpp"


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
        template<
            typename T_Species,
            typename T_GetLowerMargin = GetLowerMargin<GetPusher<bmpl::_1>>,
            typename T_GetUpperMargin = GetUpperMargin<GetPusher<bmpl::_1>>>
        struct GetMarginPusher
        {
            using AddLowerMargins = pmacc::math::CT::add<GetLowerMargin<GetInterpolation<bmpl::_1>>, T_GetLowerMargin>;
            using LowerMargin = typename bmpl::apply<AddLowerMargins, T_Species>::type;

            using AddUpperMargins = pmacc::math::CT::add<GetUpperMargin<GetInterpolation<bmpl::_1>>, T_GetUpperMargin>;
            using UpperMargin = typename bmpl::apply<AddUpperMargins, T_Species>::type;
        };

        /** Get lower margin of a pusher for species
         *
         * @tparam T_Species particle species type
         */
        template<typename T_Species>
        struct GetLowerMarginPusher
        {
            using type = typename traits::GetMarginPusher<T_Species>::LowerMargin;
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

        /** Get upper margin of a pusher for species
         *
         * @tparam T_Species particle species type
         */
        template<typename T_Species>
        struct GetUpperMarginPusher
        {
            using type = typename traits::GetMarginPusher<T_Species>::UpperMargin;
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

    } // namespace traits
} // namespace picongpu
