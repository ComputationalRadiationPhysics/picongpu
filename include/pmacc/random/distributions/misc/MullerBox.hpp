/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/algorithms/math.hpp"
#include "pmacc/random/distributions/Uniform.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace random
    {
        namespace distributions
        {
            /** create a normal distributed random number
             *
             * Create a random number with mean 0 and standard deviation 1.
             * The implementation based on the Wikipedia article:
             *    - source: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
             *    - date: 01/12/2017
             */
            template<typename T_Type, typename T_RNGMethod>
            class MullerBox : Uniform<uniform::ExcludeZero<T_Type>, T_RNGMethod>
            {
                /** The muller box is creating two random number, each second time
                 * this number is valid and can be used.
                 */
                T_Type secondRngNumber;
                //! true if secondRngNumber is valid else false
                bool hasSecondRngNumber = false;

                using RNGMethod = T_RNGMethod;
                using UniformRng = Uniform<uniform::ExcludeZero<T_Type>, RNGMethod>;
                using StateType = typename RNGMethod::StateType;

                /** generate a normal distributed random number
                 *
                 * @param acc alpaka accelerator
                 * @param state the state of an pmacc random number generator
                 */
                template<typename T_Worker>
                DINLINE T_Type getNormal(T_Worker const& worker, StateType& state)
                {
                    constexpr T_Type valueTwoPI = 6.2831853071795860;

                    T_Type u1 = UniformRng::operator()(worker, state);
                    T_Type u2 = UniformRng::operator()(worker, state) * valueTwoPI;

                    T_Type s = pmacc::math::sqrt(T_Type(-2.0) * pmacc::math::log(u1));

                    T_Type firstRngNumber;
                    pmacc::math::sincos(u2, firstRngNumber, secondRngNumber);

                    firstRngNumber *= s;
                    secondRngNumber *= s;
                    hasSecondRngNumber = true;
                    return firstRngNumber;
                }

            public:
                //! result type of the random number
                using result_type = T_Type;

                /** generate a normal distributed random number
                 *
                 * Generates two random numbers with the first call, each second call
                 * the precomputed random number is returned.
                 *
                 * @param acc alpaka accelerator
                 * @param state the state of an pmacc random number generator
                 */
                template<typename T_Worker>
                DINLINE result_type operator()(T_Worker const& worker, StateType& state)
                {
                    T_Type result;
                    if(hasSecondRngNumber)
                    {
                        result = secondRngNumber;
                        hasSecondRngNumber = false;
                    }
                    else
                    {
                        result = getNormal(worker, state);
                    }
                    return result;
                }
            };

        } // namespace distributions
    } // namespace random
} // namespace pmacc
