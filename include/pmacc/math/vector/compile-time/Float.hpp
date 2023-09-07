/* Copyright 2013-2023 Heiko Burau, Rene Widera
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

#include <cstdint>

namespace pmacc
{
    namespace math
    {
        namespace CT
        {
            template<typename X = void, typename Y = void, typename Z = void>
            struct Float
            {
                using x = X;
                using y = Y;
                using z = Z;

                static constexpr std::uint32_t dim = 3u;
            };

            template<>
            struct Float<>
            {
            };

            template<typename X>
            struct Float<X>
            {
                using x = X;

                static constexpr std::uint32_t dim = 1u;
            };

            template<typename X, typename Y>
            struct Float<X, Y>
            {
                using x = X;
                using y = Y;

                static constexpr std::uint32_t dim = 2u;
            };

        } // namespace CT
    } // namespace math
} // namespace pmacc
