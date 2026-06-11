/*
 * SPDX-FileCopyrightText: Alexander Grund, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/random/distributions/Uniform.hpp"
#include "pmacc/random/distributions/uniform/Range.hpp"
#include "pmacc/random/distributions/uniform/Uniform_double.hpp"
#include "pmacc/random/distributions/uniform/Uniform_float.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace random
    {
        namespace distributions
        {
            namespace detail
            {
                /** Returns a random floating point value uniformly distributed in [0,1)
                 *
                 * Equivalent to uniform::ExcludeOne< T_Type >::Reduced
                 */
                template<typename T_Type, class T_RNGMethod>
                class Uniform<T_Type, T_RNGMethod, std::enable_if_t<std::is_floating_point_v<T_Type>>>
                    : public distributions::Uniform<typename uniform::ExcludeOne<T_Type>::Reduced, T_RNGMethod>
                {
                };

                /** Returns a random floating point value uniformly distributed in [0,1)
                 *
                 * Equivalent to uniform::ExcludeOne< T_Type >::Reduced
                 */
                template<typename T_Type, class T_RNGMethod>
                class Uniform<
                    uniform::ExcludeOne<T_Type>,
                    T_RNGMethod,
                    std::enable_if_t<std::is_floating_point_v<T_Type>>>
                    : public distributions::Uniform<typename uniform::ExcludeOne<T_Type>::Reduced, T_RNGMethod>
                {
                };
            } // namespace detail
        } // namespace distributions
    } // namespace random
} // namespace pmacc
