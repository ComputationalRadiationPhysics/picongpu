/* Copyright 2015-2017 Alexander Grund, Rene Widera
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

namespace pmacc
{
namespace random
{
namespace distributions
{
namespace detail
{

    /**
     * Returns a random float value in [0,1) with normal distribution
     */
    template< typename T_RNGMethod>
    class Normal<float, T_RNGMethod, void>
    {
        typedef T_RNGMethod RNGMethod;
        typedef typename RNGMethod::StateType StateType;
    public:
        using result_type = float;

        template< typename T_Acc >
        DINLINE result_type
        operator()(
            T_Acc const & acc,
            StateType& state
        )
        {
            return RNGMethod().getNormal(
                acc,
                state
            );
        }
    };

}  // namespace detail
}  // namespace distributions
}  // namespace random
}  // namespace pmacc
