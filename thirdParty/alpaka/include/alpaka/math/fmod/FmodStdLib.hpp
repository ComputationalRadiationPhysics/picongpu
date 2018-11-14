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

#include <alpaka/math/fmod/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library fmod.
        class FmodStdLib
        {
        public:
            using FmodBase = FmodStdLib;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library fmod trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Fmod<
                FmodStdLib,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_arithmetic<Tx>::value
                    && std::is_arithmetic<Ty>::value>::type>
            {
                ALPAKA_FN_HOST static auto fmod(
                    FmodStdLib const & fmod,
                    Tx const & x,
                    Ty const & y)
                -> decltype(std::fmod(x, y))
                {
                    alpaka::ignore_unused(fmod);
                    return std::fmod(x, y);
                }
            };
        }
    }
}
