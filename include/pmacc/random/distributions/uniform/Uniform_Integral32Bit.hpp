/* Copyright 2015-2023 Alexander Grund
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

#include "pmacc/random/distributions/Uniform.hpp"
#include "pmacc/types.hpp"

#include <type_traits>

namespace pmacc
{
    namespace random
    {
        namespace distributions
        {
            namespace detail
            {
                /**
                 * Returns a random, uniformly distributed (up to) 32 bit integral value
                 */
                template<typename T_Type, class T_RNGMethod>
                class Uniform<
                    T_Type,
                    T_RNGMethod,
                    std::conditional_t<std::is_integral_v<T_Type> && sizeof(T_Type) <= 4, void, T_Type>>
                {
                    using RNGMethod = T_RNGMethod;
                    using StateType = typename RNGMethod::StateType;

                public:
                    using result_type = T_Type;

                    template<typename T_Worker>
                    DINLINE result_type operator()(T_Worker const& worker, StateType& state)
                    {
                        return static_cast<result_type>(RNGMethod().get32Bits(worker, state));
                    }
                };

            } // namespace detail
        } // namespace distributions
    } // namespace random
} // namespace pmacc
