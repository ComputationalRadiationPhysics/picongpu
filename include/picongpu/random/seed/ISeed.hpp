/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/random/seed/Seed.hpp"

#include <cstdint>

namespace picongpu
{
    namespace random
    {
        namespace seed
        {
            /** seed generator interface wrapper
             *
             * Generated seed is equal on all ranks and can be used together with an
             * rank unique seed to initialize a random number generator.
             * Depending of the generator T_SeedFunctor the seed is reproducible or
             * or changed with each program execution.
             */
            template<typename T_SeedFunctor = seed::Value<42>>
            struct ISeed
            {
                uint32_t operator()() const
                {
                    return T_SeedFunctor{}();
                }
            };
        } // namespace seed
    } // namespace random
} // namespace picongpu
