/* Copyright 2013-2021 Axel Huebl
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

namespace picongpu
{
    namespace traits
    {
        /** Convert a Splash CollectionType to a PIConGPU Type
         *
         * \tparam T_SplashType Splash CollectionType
         * \return \p ::type as public typedef
         */
        template<typename T_SplashType>
        struct SplashToPIC;

    } // namespace traits

} // namespace picongpu

#include "SplashToPIC.tpp"
