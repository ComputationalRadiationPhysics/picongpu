/*
 * SPDX-FileCopyrightText: Alexander Grund, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/random/distributions/Normal.hpp"
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
                //! Returns a normally distributed floating point with value with mean 0.0 and standard deviation 1.0
                template<typename T_Type, typename T_RNGMethod>
                class Normal<T_Type, T_RNGMethod, void>
                {
                    using RNGMethod = T_RNGMethod;
                    using StateType = typename RNGMethod::StateType;

                public:
                    using result_type = T_Type;

                    template<typename T_Worker>
                    DINLINE result_type operator()(T_Worker const& worker, StateType& state)
                    {
                        return ::alpaka::rand::distribution::createNormalReal<T_Type>(worker.getAcc())(state);
                    }
                };

            } // namespace detail
        } // namespace distributions
    } // namespace random
} // namespace pmacc
