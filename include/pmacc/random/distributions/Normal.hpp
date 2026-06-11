/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/random/methods/RngPlaceholder.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace random
    {
        namespace distributions
        {
            namespace detail
            {
                /** Only this must be specialized for different types */
                template<typename T_Type, class T_RNGMethod, class T_SFINAE = void>
                class Normal;
            } // namespace detail

            /**
             * Returns a random, normal distributed value of the given type
             */
            template<typename T_Type, class T_RNGMethod = methods::RngPlaceholder>
            struct Normal : public detail::Normal<T_Type, T_RNGMethod>
            {
                template<typename T_Method>
                struct applyMethod
                {
                    using type = Normal<T_Type, T_Method>;
                };
            };

        } // namespace distributions
    } // namespace random
} // namespace pmacc

#include "pmacc/random/distributions/normal/Normal_double.hpp"
#include "pmacc/random/distributions/normal/Normal_float.hpp"
#include "pmacc/random/distributions/normal/Normal_generic.hpp"
