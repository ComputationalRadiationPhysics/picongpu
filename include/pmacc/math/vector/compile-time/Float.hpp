/* Copyright 2013-2021 Heiko Burau, Rene Widera
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

#include <stdint.h>
#include <boost/mpl/void.hpp>

namespace mpl = boost::mpl;

namespace pmacc
{
    namespace math
    {
        namespace CT
        {
            template<typename X = mpl::void_, typename Y = mpl::void_, typename Z = mpl::void_>
            struct Float
            {
                using x = X;
                using y = Y;
                using z = Z;

                static constexpr int dim = 3;
            };

            template<>
            struct Float<>
            {
            };

            template<typename X>
            struct Float<X>
            {
                using x = X;

                static constexpr int dim = 1;
            };

            template<typename X, typename Y>
            struct Float<X, Y>
            {
                using x = X;
                using y = Y;

                static constexpr int dim = 2u;
            };

        } // namespace CT
    } // namespace math
} // namespace pmacc
