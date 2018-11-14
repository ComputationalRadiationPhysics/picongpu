/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/math/min/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>
#include <cmath>
#include <algorithm>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library min.
        class MinStdLib
        {
        public:
            using MinBase = MinStdLib;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library integral min trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Min<
                MinStdLib,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_integral<Tx>::value
                    && std::is_integral<Ty>::value>::type>
            {
                ALPAKA_FN_HOST static auto min(
                    MinStdLib const & min,
                    Tx const & x,
                    Ty const & y)
                -> decltype(std::min(x, y))
                {
                    alpaka::ignore_unused(min);
                    return std::min(x, y);
                }
            };
            //#############################################################################
            //! The standard library mixed integral floating point min trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Min<
                MinStdLib,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_arithmetic<Tx>::value
                    && std::is_arithmetic<Ty>::value
                    && !(std::is_integral<Tx>::value
                        && std::is_integral<Ty>::value)>::type>
            {
                ALPAKA_FN_HOST static auto min(
                    MinStdLib const & min,
                    Tx const & x,
                    Ty const & y)
                -> decltype(std::fmin(x, y))
                {
                    alpaka::ignore_unused(min);
                    return std::fmin(x, y);
                }
            };
        }
    }
}
