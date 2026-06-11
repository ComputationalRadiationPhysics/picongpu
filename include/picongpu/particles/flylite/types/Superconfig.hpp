/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/math/Vector.hpp>
#include <pmacc/types.hpp>

namespace picongpu
{
    namespace flylite
    {
        namespace types
        {
            /** Ion Superconfiguration
             *
             * This is the attribute type for an ion's screened hydrogenic
             * superconfiguration.
             *
             * See for details on screened hydrogenic levels:
             *   H.-K. Chung, S.H. Hansen, H.A. Scott.
             *   *Generalized Collisional Radiative Model Using*
             *   *Screened Hydrogenic Levels*,
             *   in Modern Methods in Collisional-Radiative Modeling of Plasmas,
             *   edited by Y. Ralchenko (Springer, 2016) pp.51-79
             *
             * @tparam T_Type the float type to use, e.g. float_64
             * @tparam T_populations the number of populations to store for each ion,
             *                       range: [0, 255]
             */
            template<typename T_Type, uint8_t T_populations>
            using Superconfig = pmacc::math::Vector<T_Type, T_populations>;
        } // namespace types
    } // namespace flylite
} // namespace picongpu
