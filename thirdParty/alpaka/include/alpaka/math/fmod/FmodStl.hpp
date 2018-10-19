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

#include <boost/core/ignore_unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library fmod.
        class FmodStl
        {
        public:
            using FmodBase = FmodStl;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library fmod trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Fmod<
                FmodStl,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_arithmetic<Tx>::value
                    && std::is_arithmetic<Ty>::value>::type>
            {
                ALPAKA_FN_ACC_NO_CUDA static auto fmod(
                    FmodStl const & fmod,
                    Tx const & x,
                    Ty const & y)
                -> decltype(std::fmod(x, y))
                {
                    boost::ignore_unused(fmod);
                    return std::fmod(x, y);
                }
            };
        }
    }
}
