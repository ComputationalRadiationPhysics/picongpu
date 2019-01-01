/* Copyright 2013-2019 Heiko Burau, Rene Widera
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
namespace nvidia
{
namespace rng
{
namespace distributions
{
namespace detail
{
    /*Return normally distributed floats with mean 0.0f and standard deviation 1.0f
     */
    template< typename T_Acc>
    class Normal_float
    {
    public:
        typedef float Type;
    private:
        using Dist =
            decltype(
                ::alpaka::rand::distribution::createNormalReal<Type>(
                    alpaka::core::declval<T_Acc const &>()));
        PMACC_ALIGN(dist, Dist);
    public:
        HDINLINE Normal_float()
        {
        }

        HDINLINE Normal_float(const T_Acc& acc) : dist(::alpaka::rand::distribution::createNormalReal<Type>(acc))
        {
        }

        template<class RNGState>
        DINLINE Type operator()(RNGState& state)
        {
            return dist(state);
        }

    };
} // namespace detail

    struct Normal_float
    {
        template< typename T_Acc>
        static HDINLINE detail::Normal_float< T_Acc >
        get( T_Acc const & acc)
        {
            return detail::Normal_float< T_Acc >( acc );
        }
    };
} // namespace distributions
} // namespace rng
} // namespace nvidia
} // namespace pmacc
