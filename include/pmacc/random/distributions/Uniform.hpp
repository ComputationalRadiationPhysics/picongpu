/* Copyright 2015-2021 Alexander Grund
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
#include "pmacc/random/methods/RngPlaceholder.hpp"
#include "pmacc/random/distributions/uniform/Range.hpp"

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

#include "pmacc/random/distributions/uniform/Uniform_float.hpp"
#include "pmacc/random/distributions/uniform/Uniform_double.hpp"
#include "pmacc/random/distributions/uniform/Uniform_generic.hpp"
#include "pmacc/random/distributions/uniform/Uniform_Integral32Bit.hpp"
#include "pmacc/random/distributions/uniform/Uniform_Integral64Bit.hpp"
