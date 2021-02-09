/* Copyright 2013-2021 Marco Garten, Heiko Burau, Rene Widera, Richard Pausch
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

/** \file
 *
 * Calculation utilities to be relocated together with `plugins/radiation/utilities.hpp`
 * to a more appropriate place in a more generic pmacc-y way
 */

namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            namespace util
            {
                /* power 2 function */
                template<typename A>
                HDINLINE A square(A a)
                {
                    return a * a;
                }
                /* power 2 function with different result type */
                template<typename A, typename R>
                HDINLINE R square(A a)
                {
                    return a * a;
                }
                /* power 3 function */
                template<typename A>
                HDINLINE A cube(A a)
                {
                    return a * a * a;
                }
                /* power 3 function with different result type */
                template<typename A, typename R>
                HDINLINE R cube(A a)
                {
                    return a * a * a;
                }
                /* power 4 function */
                template<typename A>
                HDINLINE A quad(A a)
                {
                    const A b = a * a;
                    return b * b;
                }
                /* power 4 function with different result type */
                template<typename A, typename R>
                HDINLINE R quad(A a)
                {
                    const R b = a * a;
                    return b * b;
                }

            } // namespace util

        } // namespace ionization
    } // namespace particles
} // namespace picongpu
