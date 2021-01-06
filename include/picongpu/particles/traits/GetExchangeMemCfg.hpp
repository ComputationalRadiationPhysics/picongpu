/* Copyright 2017-2021 Rene Widera
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

#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/traits/HasFlag.hpp>

#include <boost/mpl/if.hpp>


namespace picongpu
{
    namespace traits
    {
        /** get a memory configuration for species exchange buffer
         *
         * If exchangeMemCfg is not defined for a species than the default memory
         * exchange size from the file memory.param are used.
         *
         * @tparam T_Species picongpu::Particles, type of the species
         * @return class with buffer sizes for each direction
         */
        template<typename T_Species>
        struct GetExchangeMemCfg
        {
            using FrameType = typename T_Species::FrameType;
            using hasMemCfg = typename HasFlag<FrameType, exchangeMemCfg<>>::type;

            using type = typename bmpl::if_<
                hasMemCfg,
                typename pmacc::traits::Resolve<typename GetFlagType<FrameType, exchangeMemCfg<>>::type>::type,
                ::picongpu::DefaultExchangeMemCfg>::type;
        };

        //! short hand traits for GetExchangeMemCfg
        template<typename T_Species>
        using GetExchangeMemCfg_t = typename traits::GetExchangeMemCfg<T_Species>::type;

    } // namespace traits
} // namespace picongpu
