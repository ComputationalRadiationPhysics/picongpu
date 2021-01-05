/* Copyright 2015-2021 Alexander Grund, Rene Widera
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
#include "pmacc/random/distributions/Uniform.hpp"
#include "pmacc/random/distributions/uniform/Uniform_float.hpp"
#include "pmacc/random/distributions/uniform/Uniform_double.hpp"
#include "pmacc/random/distributions/uniform/Range.hpp"

namespace pmacc
{
    namespace random
    {
        namespace distributions
        {
            namespace detail
            {
                /** Returns a random floating point value uniformly distributed in [0,1)
                 *
                 * Equivalent to uniform::ExcludeOne< T_Type >::Reduced
                 */
                template<typename T_Type, class T_RNGMethod>
                class Uniform<
                    T_Type,
                    T_RNGMethod,
                    typename std::enable_if<std::is_floating_point<T_Type>::value>::type>
                    : public pmacc::random::distributions::
                          Uniform<typename uniform::ExcludeOne<T_Type>::Reduced, T_RNGMethod>
                {
                };

                /** Returns a random floating point value uniformly distributed in [0,1)
                 *
                 * Equivalent to uniform::ExcludeOne< T_Type >::Reduced
                 */
                template<typename T_Type, class T_RNGMethod>
                class Uniform<
                    uniform::ExcludeOne<T_Type>,
                    T_RNGMethod,
                    typename std::enable_if<std::is_floating_point<T_Type>::value>::type>
                    : public pmacc::random::distributions::
                          Uniform<typename uniform::ExcludeOne<T_Type>::Reduced, T_RNGMethod>
                {
                };
            } // namespace detail
        } // namespace distributions
    } // namespace random
} // namespace pmacc
