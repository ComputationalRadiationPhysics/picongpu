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

#include <alpaka/math/cbrt/Traits.hpp>

#include <boost/core/ignore_unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library cbrt.
        class CbrtStl
        {
        public:
            using CbrtBase = CbrtStl;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library cbrt trait specialization.
            template<
                typename TArg>
            struct Cbrt<
                CbrtStl,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                ALPAKA_FN_ACC_NO_CUDA static auto cbrt(
                    CbrtStl const & cbrt,
                    TArg const & arg)
                -> decltype(std::cbrt(arg))
                {
                    boost::ignore_unused(cbrt);
                    return std::cbrt(arg);
                }
            };
        }
    }
}
