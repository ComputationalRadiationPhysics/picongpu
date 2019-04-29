/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera
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
    /*create a random float number from [0.0,1.0)
     */
    template< typename T_Acc>
    class Uniform_float
    {
    public:
        typedef float Type;
    private:
        using Dist =
            decltype(
                ::alpaka::rand::distribution::createUniformReal<Type>(
                    alpaka::core::declval<T_Acc const &>()));
        PMACC_ALIGN(dist, Dist);
    public:

        HDINLINE Uniform_float()
        {
        }

        HDINLINE Uniform_float(const T_Acc& acc) : dist(::alpaka::rand::distribution::createUniformReal<Type>(acc))
        {
        }

        template<class RNGState>
        DINLINE Type operator()(RNGState& state)
        {
            // (0.f, 1.0f]
            const Type raw = dist(state);

            /// \warn hack, are is that really ok? I say, yes, since
            /// it shifts just exactly one number. Axel
            ///
            ///   Note: (1.0f - raw) does not work, since
            ///         nvidia seems to return denormalized
            ///         floats around 0.f (thats not as they
            ///         state it out in their documentation)
            // [0.f, 1.0f)
            const Type r = raw * static_cast<float>( raw != Type(1.0) );
            return r;
        }

    };
} // namespace detail

    struct Uniform_float
    {
        template< typename T_Acc>
        static HDINLINE detail::Uniform_float< T_Acc >
        get( T_Acc const & acc)
        {
            return detail::Uniform_float< T_Acc >( acc );
        }
    };
} // namespace distributions
} // namespace rng
} // namespace nvidia
} // namespace pmacc
