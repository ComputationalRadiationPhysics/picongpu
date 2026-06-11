/*
 * SPDX-FileCopyrightText: Alexander Grund, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/random/distributions/Uniform.hpp"
#include "pmacc/random/distributions/uniform/Range.hpp"
#include "pmacc/types.hpp"

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
                    using RNGMethod = T_RNGMethod;
                    using StateType = typename RNGMethod::StateType;
                    using result_type = float;

                    template<typename T_Worker>
                    DINLINE float operator()(T_Worker const& worker, StateType& state) const
                    {
                        float const value2pow32Inv = 2.3283064e-10f;
                        uint32_t const random = RNGMethod().get32Bits(worker, state);
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
                    using RNGMethod = T_RNGMethod;
                    using StateType = typename RNGMethod::StateType;
                    using result_type = float;

                    template<typename T_Worker>
                    DINLINE float operator()(T_Worker const& worker, StateType& state) const
                    {
                        float const randomValue
                            = pmacc::random::distributions::Uniform<uniform::ExcludeZero<float>, RNGMethod>()(
                                worker,
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
                    using RNGMethod = T_RNGMethod;
                    using StateType = typename RNGMethod::StateType;
                    using result_type = float;

                    template<typename T_Worker>
                    DINLINE float operator()(T_Worker const& worker, StateType& state) const
                    {
                        float const value2pow24Inv = 5.9604645e-08f;
                        float const randomValue24Bit = RNGMethod().get32Bits(worker, state) >> 8;
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
                    using RNGMethod = T_RNGMethod;
                    using StateType = typename RNGMethod::StateType;
                    using result_type = float;

                    template<typename T_Worker>
                    DINLINE float operator()(T_Worker const& worker, StateType& state) const
                    {
                        do
                        {
                            float const randomValue
                                = pmacc::random::distributions::Uniform<uniform::ExcludeZero<float>, RNGMethod>()(
                                    worker,
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
