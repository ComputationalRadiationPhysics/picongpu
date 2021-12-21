/* Copyright 2016-2021 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
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
