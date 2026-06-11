/*
 * SPDX-FileCopyrightText: Alexander Grund, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
                 * Returns a random, uniformly distributed (up to) 64 bit integral value
                 */
                template<typename T_Type, class T_RNGMethod>
                class Uniform<
                    T_Type,
                    T_RNGMethod,
                    std::conditional_t<std::is_integral_v<T_Type> && sizeof(T_Type) == 8, void, T_Type>>
                {
                    using RNGMethod = T_RNGMethod;
                    using StateType = typename RNGMethod::StateType;

                public:
                    using result_type = T_Type;

                    template<typename T_Worker>
                    DINLINE result_type operator()(T_Worker const& worker, StateType& state)
                    {
                        return static_cast<result_type>(RNGMethod().get64Bits(worker, state));
                    }
                };

            } // namespace detail
        } // namespace distributions
    } // namespace random
} // namespace pmacc
