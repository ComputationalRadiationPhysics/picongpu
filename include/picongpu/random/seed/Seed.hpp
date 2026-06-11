/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
