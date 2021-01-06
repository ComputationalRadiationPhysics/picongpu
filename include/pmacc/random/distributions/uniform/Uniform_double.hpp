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
                /** Returns a random double value uniformly distributed in (0,1]
                 *
                 * The smallest created value is `2^-65` (~ `2.710505431213761*10^-20`)
                 */
                template<class T_RNGMethod>
                class Uniform<uniform::ExcludeZero<double>, T_RNGMethod, void>
                {
                public:
                    typedef T_RNGMethod RNGMethod;
                    typedef typename RNGMethod::StateType StateType;
                    typedef double result_type;

                    template<typename T_Acc>
                    DINLINE double operator()(T_Acc const& acc, StateType& state) const
                    {
                        double const value2pow64Inv = 5.421010862427522e-20;
                        uint64_t const random = RNGMethod().get64Bits(acc, state);
                        return static_cast<double>(random) * value2pow64Inv + (value2pow64Inv / 2.0);
                    }
                };

                /** Returns a random double value uniformly distributed in [0,1)
                 *
                 * Swap the value one to zero (creates a small error in uniform distribution)
                 */
                template<class T_RNGMethod>
                class Uniform<uniform::ExcludeOne<double>::SwapOneToZero, T_RNGMethod, void>
                {
                public:
                    typedef T_RNGMethod RNGMethod;
                    typedef typename RNGMethod::StateType StateType;
                    typedef double result_type;

                    template<typename T_Acc>
                    DINLINE double operator()(T_Acc const& acc, StateType& state) const
                    {
                        double const randomValue
                            = pmacc::random::distributions::Uniform<uniform::ExcludeZero<double>, RNGMethod>()(
                                acc,
                                state);
                        return randomValue == 1.0 ? 0.0 : randomValue;
                    }
                };

                /** Returns a random double value uniformly distributed in [0,1)
                 *
                 * Number of unique random numbers is reduced to `2^53`.
                 * Uses a uniform distance of `2^-53` (`epsilon/2`) between each possible
                 * random number.
                 */
                template<class T_RNGMethod>
                class Uniform<uniform::ExcludeOne<double>::Reduced, T_RNGMethod, void>
                {
                public:
                    typedef T_RNGMethod RNGMethod;
                    typedef typename RNGMethod::StateType StateType;
                    typedef double result_type;

                    template<typename T_Acc>
                    DINLINE double operator()(T_Acc const& acc, StateType& state) const
                    {
                        double const value2pow53Inv = 1.1102230246251565e-16;
                        double const randomValue53Bit = RNGMethod().get64Bits(acc, state) >> 11;
                        return randomValue53Bit * value2pow53Inv;
                    }
                };

                /** Returns a random double value uniformly distributed in (0,1)
                 *
                 * Loops until a random value inside the defined range is created.
                 * The runtime of this method is not deterministic.
                 */
                template<class T_RNGMethod>
                class Uniform<typename uniform::ExcludeOne<double>::Repeat, T_RNGMethod, void>
                {
                public:
                    typedef T_RNGMethod RNGMethod;
                    typedef typename RNGMethod::StateType StateType;
                    typedef double result_type;

                    template<typename T_Acc>
                    DINLINE result_type operator()(T_Acc const& acc, StateType& state) const
                    {
                        do
                        {
                            const double randomValue
                                = pmacc::random::distributions::Uniform<uniform::ExcludeZero<double>, RNGMethod>()(
                                    acc,
                                    state);

                            if(randomValue != 1.0)
                                return randomValue;
                        } while(true);
                    }
                };

            } // namespace detail
        } // namespace distributions
    } // namespace random
} // namespace pmacc
