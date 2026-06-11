/*
 * SPDX-FileCopyrightText: Richard Pausch, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
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
        template<
            typename T_Species,
            typename T_GetLowerMargin = traits::GetLowerMargin<traits::GetPusher<boost::mpl::_1>>,
            typename T_GetUpperMargin = traits::GetUpperMargin<traits::GetPusher<boost::mpl::_1>>>
        struct GetMarginPusher
        {
            using AddLowerMargins = pmacc::math::CT::
                add<traits::GetLowerMargin<traits::GetInterpolation<boost::mpl::_1>>, T_GetLowerMargin>;
            using LowerMargin = typename boost::mpl::apply<AddLowerMargins, T_Species>::type;

            using AddUpperMargins
                = pmacc::math::CT::add<GetUpperMargin<traits::GetInterpolation<boost::mpl::_1>>, T_GetUpperMargin>;
            using UpperMargin = typename boost::mpl::apply<AddUpperMargins, T_Species>::type;
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
                typename traits::GetLowerMargin<T_Pusher>::type,
                typename traits::GetUpperMargin<T_Pusher>::type>::LowerMargin;
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
                typename traits::GetUpperMargin<T_Pusher>::type>::UpperMargin;
        };

    } // namespace traits
} // namespace picongpu
