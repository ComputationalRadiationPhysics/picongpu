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

#include <alpaka/math/pow/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library pow.
        class PowStdLib
        {
        public:
            using PowBase = PowStdLib;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library pow trait specialization.
            template<
                typename TBase,
                typename TExp>
            struct Pow<
                PowStdLib,
                TBase,
                TExp,
                typename std::enable_if<
                    std::is_arithmetic<TBase>::value
                    && std::is_arithmetic<TExp>::value>::type>
            {
                ALPAKA_FN_HOST static auto pow(
                    PowStdLib const & pow,
                    TBase const & base,
                    TExp const & exp)
                -> decltype(std::pow(base, exp))
                {
                    alpaka::ignore_unused(pow);
                    return std::pow(base, exp);
                }
            };
        }
    }
}
