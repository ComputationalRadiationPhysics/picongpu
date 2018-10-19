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

#include <boost/core/ignore_unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library pow.
        class PowStl
        {
        public:
            using PowBase = PowStl;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library pow trait specialization.
            template<
                typename TBase,
                typename TExp>
            struct Pow<
                PowStl,
                TBase,
                TExp,
                typename std::enable_if<
                    std::is_arithmetic<TBase>::value
                    && std::is_arithmetic<TExp>::value>::type>
            {
                ALPAKA_FN_ACC_NO_CUDA static auto pow(
                    PowStl const & pow,
                    TBase const & base,
                    TExp const & exp)
                -> decltype(std::pow(base, exp))
                {
                    boost::ignore_unused(pow);
                    return std::pow(base, exp);
                }
            };
        }
    }
}
