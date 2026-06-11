/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    namespace random
    {
        namespace distributions
        {
            namespace uniform
            {
                /** floating point number in the range (0,1]
                 *
                 * @tparam T_Type type of the result
                 * @return value in the range (0,1]
                 */
                template<typename T_Type>
                struct ExcludeZero
                {
                };

                /**  floating point number in the range [0,1)
                 *
                 * @tparam T_Type type of the result
                 */
                template<typename T_Type>
                struct ExcludeOne
                {
                    /** Reduce the random range
                     *
                     * number of unique random numbers for
                     *   - `float` is `2^24`
                     *   - `double` is `2^53`
                     *
                     * Creates intervals with the width of epsilon/2.
                     */
                    struct Reduced
                    {
                    };

                    /** Loops until a random value inside the defined range is created
                     *
                     * The runtime of this method is not deterministic.
                     * @warning zero is excluded which results in a range (0,1)
                     */
                    struct Repeat
                    {
                    };

                    /** Swap the value one to zero
                     *
                     * This method creates a small error in uniform distribution
                     */
                    struct SwapOneToZero
                    {
                    };
                };

            } // namespace uniform
        } // namespace distributions
    } // namespace random
} // namespace pmacc
