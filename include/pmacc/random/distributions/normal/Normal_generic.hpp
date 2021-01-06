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
#include "pmacc/random/distributions/Normal.hpp"

#include <type_traits>


namespace pmacc
{
    namespace random
    {
        namespace distributions
        {
            namespace detail
            {
                //! Returns a normally distributed floating point with value with mean 0.0 and standard deviation 1.0
                template<typename T_Type, typename T_RNGMethod>
                class Normal<T_Type, T_RNGMethod, void>
                {
                    using RNGMethod = T_RNGMethod;
                    using StateType = typename RNGMethod::StateType;

                public:
                    using result_type = T_Type;

                    template<typename T_Acc>
                    DINLINE result_type operator()(T_Acc const& acc, StateType& state)
                    {
                        return ::alpaka::rand::distribution::createNormalReal<T_Type>(acc)(state);
                    }
                };

            } // namespace detail
        } // namespace distributions
    } // namespace random
} // namespace pmacc
