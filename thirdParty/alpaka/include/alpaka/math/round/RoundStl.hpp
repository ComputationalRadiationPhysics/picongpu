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

#include <alpaka/math/round/Traits.hpp>

#include <boost/core/ignore_unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library round.
        class RoundStl
        {
        public:
            using RoundBase = RoundStl;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library round trait specialization.
            template<
                typename TArg>
            struct Round<
                RoundStl,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                ALPAKA_FN_ACC_NO_CUDA static auto round(
                    RoundStl const & round,
                    TArg const & arg)
                -> decltype(std::round(arg))
                {
                    boost::ignore_unused(round);
                    return std::round(arg);
                }
            };
            //#############################################################################
            //! The standard library round trait specialization.
            template<
                typename TArg>
            struct Lround<
                RoundStl,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                ALPAKA_FN_ACC_NO_CUDA static auto lround(
                    RoundStl const & lround,
                    TArg const & arg)
                -> long int
                {
                    boost::ignore_unused(lround);
                    return std::lround(arg);
                }
            };
            //#############################################################################
            //! The standard library round trait specialization.
            template<
                typename TArg>
            struct Llround<
                RoundStl,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                ALPAKA_FN_ACC_NO_CUDA static auto llround(
                    RoundStl const & llround,
                    TArg const & arg)
                -> long int
                {
                    boost::ignore_unused(llround);
                    return std::llround(arg);
                }
            };
        }
    }
}
