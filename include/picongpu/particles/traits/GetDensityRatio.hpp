/*
 * SPDX-FileCopyrightText: Rene Widera, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/Resolve.hpp>

namespace picongpu
{
    namespace traits
    {
        namespace detail
        {
            value_identifier(float_X, DefaultDensityRatio, 1.0);
        } // namespace detail

        /** get density ratio of a species
         *
         * ratio is set to 1.0 if no alias `densityRatio<>` is defined
         *
         * @treturn ::type `value_identifier` with the default density
         */
        template<typename T_Species>
        struct GetDensityRatio
        {
            using FrameType = typename T_Species::FrameType;
            using hasDensityRatio = typename HasFlag<FrameType, densityRatio<>>::type;
            using DensityRatioOfSpecies = typename pmacc::traits::Resolve<
                typename pmacc::traits::GetFlagType<FrameType, densityRatio<>>::type>::type;

            using type = pmacc::mp_if<hasDensityRatio, DensityRatioOfSpecies, detail::DefaultDensityRatio>;
        };

    } // namespace traits
} // namespace picongpu
