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

#include <alpaka/math/abs/Traits.hpp>   // Abs

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <type_traits>                  // std::enable_if, std::is_arithmetic, std::is_signed
#include <cstdlib>                      // std::abs(int)
#include <cmath>                        // std::abs(float)

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library abs.
        //#############################################################################
        class AbsStl
        {
        public:
            using AbsBase = AbsStl;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library abs trait specialization.
            //#############################################################################
            template<
                typename TArg>
            struct Abs<
                AbsStl,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value
                    && std::is_signed<TArg>::value>::type>
            {
                ALPAKA_FN_ACC_NO_CUDA static auto abs(
                    AbsStl const & abs,
                    TArg const & arg)
                -> decltype(std::abs(arg))
                {
                    boost::ignore_unused(abs);
                    return std::abs(arg);
                }
            };
        }
    }
}
