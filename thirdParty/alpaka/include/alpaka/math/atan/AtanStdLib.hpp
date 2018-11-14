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

#include <alpaka/math/atan/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library atan.
        class AtanStdLib
        {
        public:
            using AtanBase = AtanStdLib;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library atan trait specialization.
            template<
                typename TArg>
            struct Atan<
                AtanStdLib,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                ALPAKA_FN_HOST static auto atan(
                    AtanStdLib const & atan,
                    TArg const & arg)
                -> decltype(std::atan(arg))
                {
                    alpaka::ignore_unused(atan);
                    return std::atan(arg);
                }
            };
        }
    }
}
