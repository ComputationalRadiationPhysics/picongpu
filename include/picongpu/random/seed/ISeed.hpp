/* Copyright 2018-2021 Rene Widera
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
