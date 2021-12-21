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

#include <cstdint>


namespace picongpu
{
    namespace random
    {
        namespace seed
        {
            /** constant seed
             *
             * The seed is equal on each program program start.
             */
            template<uint32_t T_constSeedValue>
            struct Value
            {
                uint32_t operator()() const
                {
                    return T_constSeedValue;
                }
            };

            /** time dependant seed
             *
             * The seed is derived from the current system time.
             * The seed is different with each program start.
             */
            struct FromTime
            {
                uint32_t operator()() const;
            };

            /** read the seed from the environment
             *
             * Read the seed from the environment variable `PIC_SEED`.
             * If `PIC_SEED` is not defined zero will be returned.
             */
            struct FromEnvironment
            {
                uint32_t operator()() const;
            };

        } // namespace seed
    } // namespace random
} // namespace picongpu
