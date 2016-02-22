/**
 * Copyright 2015-2016 Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc_types.hpp"
#include "random/distributions/Uniform.hpp"
#include <boost/type_traits.hpp>

namespace PMacc
{
namespace random
{
namespace distributions
{
namespace detail
{

    /**
     * Returns a random, uniformly distributed (up to) 32 bit integral value
     */
    template<typename T_Type, class T_RNGMethod>
    class Uniform<
        T_Type,
        T_RNGMethod,
        typename bmpl::if_c<
            boost::is_integral<T_Type>::value && sizeof(T_Type) <= 4,
            void,
            T_Type
        >::type
    >
    {
        typedef T_RNGMethod RNGMethod;
        typedef typename RNGMethod::StateType StateType;
    public:
        typedef T_Type result_type;

        DINLINE result_type
        operator()(StateType& state)
        {
            return static_cast<result_type>(RNGMethod().get32Bits(state));
        }
    };

}  // namespace detail
}  // namespace distributions
}  // namespace random
}  // namespace PMacc
