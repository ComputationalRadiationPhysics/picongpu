/*
 * SPDX-FileCopyrightText: Rene Widera
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

            using type = pmacc::mp_if<
                hasMemCfg,
                typename pmacc::traits::Resolve<
                    typename pmacc::traits::GetFlagType<FrameType, exchangeMemCfg<>>::type>::type,
                ::picongpu::DefaultExchangeMemCfg>;
        };

        //! short hand traits for GetExchangeMemCfg
        template<typename T_Species>
        using GetExchangeMemCfg_t = typename GetExchangeMemCfg<T_Species>::type;

    } // namespace traits
} // namespace picongpu
