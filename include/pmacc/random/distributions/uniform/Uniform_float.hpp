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
#include "pmacc/random/distributions/uniform/Range.hpp"

namespace pmacc
{
    namespace random
    {
        namespace distributions
        {
            namespace detail
            {
                /** Returns a random float value uniformly distributed in (0,1]
                 *
                 * The smallest created value is `2^-33` (~ `1.164*10^-10`)
                 */
                template<class T_RNGMethod>
                class Uniform<uniform::ExcludeZero<float>, T_RNGMethod, void>
                {
                public:
                    typedef T_RNGMethod RNGMethod;
                    typedef typename RNGMethod::StateType StateType;
                    typedef float result_type;

                    template<typename T_Acc>
                    DINLINE float operator()(T_Acc const& acc, StateType& state) const
                    {
                        const float value2pow32Inv = 2.3283064e-10f;
                        const uint32_t random = RNGMethod().get32Bits(acc, state);
                        return static_cast<float>(random) * value2pow32Inv + (value2pow32Inv / 2.0f);
                    }
                };

                /** Returns a random float value uniformly distributed in [0,1)
                 *
                 * Swap the value one to zero (creates a small error in uniform distribution)
                 */
                template<class T_RNGMethod>
                class Uniform<uniform::ExcludeOne<float>::SwapOneToZero, T_RNGMethod, void>
                {
                public:
                    typedef T_RNGMethod RNGMethod;
                    typedef typename RNGMethod::StateType StateType;
                    typedef float result_type;

                    template<typename T_Acc>
                    DINLINE float operator()(T_Acc const& acc, StateType& state) const
                    {
                        const float randomValue
                            = pmacc::random::distributions::Uniform<uniform::ExcludeZero<float>, RNGMethod>()(
                                acc,
                                state);
                        return randomValue == 1.0f ? 0.0f : randomValue;
                    }
                };

                /** Returns a random float value uniformly distributed in [0,1)
                 *
                 * Number of unique random numbers is reduced to `2^24`.
                 * Uses a uniform distance of `2^-24` (`epsilon/2`) between each possible
                 * random number.
                 */
                template<class T_RNGMethod>
                class Uniform<uniform::ExcludeOne<float>::Reduced, T_RNGMethod, void>
                {
                public:
                    typedef T_RNGMethod RNGMethod;
                    typedef typename RNGMethod::StateType StateType;
                    typedef float result_type;

                    template<typename T_Acc>
                    DINLINE float operator()(T_Acc const& acc, StateType& state) const
                    {
                        const float value2pow24Inv = 5.9604645e-08f;
                        const float randomValue24Bit = RNGMethod().get32Bits(acc, state) >> 8;
                        return static_cast<float>(randomValue24Bit) * value2pow24Inv;
                    }
                };

                /** Returns a random float value uniformly distributed in (0,1)
                 *
                 * Loops until a random value inside the defined range is created.
                 * The runtime of this method is not deterministic.
                 */
                template<class T_RNGMethod>
                class Uniform<typename uniform::ExcludeOne<float>::Repeat, T_RNGMethod, void>
                {
                public:
                    typedef T_RNGMethod RNGMethod;
                    typedef typename RNGMethod::StateType StateType;
                    typedef float result_type;

                    template<typename T_Acc>
                    DINLINE float operator()(T_Acc const& acc, StateType& state) const
                    {
                        do
                        {
                            const float randomValue
                                = pmacc::random::distributions::Uniform<uniform::ExcludeZero<float>, RNGMethod>()(
                                    acc,
                                    state);

                            if(randomValue != 1.0f)
                                return randomValue;
                        } while(true);
                    }
                };

            } // namespace detail
        } // namespace distributions
    } // namespace random
} // namespace pmacc
