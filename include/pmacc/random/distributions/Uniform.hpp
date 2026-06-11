/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/random/distributions/uniform/Range.hpp"
#include "pmacc/random/methods/RngPlaceholder.hpp"
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
                /** Only this must be specialized for different types */
                template<typename T_Type, class T_RNGMethod, class T_SFINAE = void>
                class Uniform;

            } // namespace detail

            /**
             * Returns a random, uniformly distributed value of the given type
             *
             * @tparam T_Type the result type or a range description @see uniform/Range.hpp
             * \code
             * Uniform<uniform::ExcludeOne<float>::Reduced> UniformReducedDistribution; //default
             * Uniform<float> UniformDefaultDistribution; //equal to line one
             * Uniform<uniform::ExcludeZero<float> > UniformNoZeroDistribution;
             * \endcode
             * @tparam T_RNGMethod method to create a random number
             */
            template<typename T_Type, class T_RNGMethod = methods::RngPlaceholder>
            struct Uniform : public detail::Uniform<T_Type, T_RNGMethod>
            {
                template<typename T_Method>
                struct applyMethod
                {
                    using type = Uniform<T_Type, T_Method>;
                };
            };

        } // namespace distributions
    } // namespace random
} // namespace pmacc

#include "pmacc/random/distributions/uniform/Uniform_Integral32Bit.hpp"
#include "pmacc/random/distributions/uniform/Uniform_Integral64Bit.hpp"
#include "pmacc/random/distributions/uniform/Uniform_double.hpp"
#include "pmacc/random/distributions/uniform/Uniform_float.hpp"
#include "pmacc/random/distributions/uniform/Uniform_generic.hpp"
