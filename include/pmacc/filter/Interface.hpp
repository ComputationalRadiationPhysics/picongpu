/* Copyright 2017-2021 Rene Widera
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
#include "pmacc/functor/Interface.hpp"


namespace pmacc
{
    namespace filter
    {
        /** Interface for a filter
         *
         * A filter is a functor which is evaluated to true or false depending
         * on the input parameters.
         * A filter can be used to decide e.g. if a particle is located in a user
         * defined area or if an attribute is above a threshold.
         *
         * @tparam T_UserFunctor pmacc::functor::Interface, type of the functor (filter rule)
         * @tparam T_numArguments number of arguments which must be supported by T_UserFunctor
         */
        template<typename T_UserFunctor, uint32_t T_numArguments>
        using Interface = pmacc::functor::Interface<T_UserFunctor, T_numArguments, bool>;

    } // namespace filter
} // namespace pmacc
