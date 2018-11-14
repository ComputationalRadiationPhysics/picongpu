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

#include <alpaka/math/abs/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>
#include <cstdlib>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library abs.
        class AbsStdLib
        {
        public:
            using AbsBase = AbsStdLib;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library abs trait specialization.
            template<
                typename TArg>
            struct Abs<
                AbsStdLib,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value
                    && std::is_signed<TArg>::value>::type>
            {
                ALPAKA_FN_HOST static auto abs(
                    AbsStdLib const & abs,
                    TArg const & arg)
                -> decltype(std::abs(arg))
                {
                    alpaka::ignore_unused(abs);
                    return std::abs(arg);
                }
            };
        }
    }
}
