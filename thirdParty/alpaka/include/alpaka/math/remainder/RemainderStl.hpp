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

#include <alpaka/math/remainder/Traits.hpp>

#include <boost/core/ignore_unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library remainder.
        class RemainderStl
        {
        public:
            using RemainderBase = RemainderStl;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library remainder trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Remainder<
                RemainderStl,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_integral<Tx>::value
                    && std::is_integral<Ty>::value>::type>
            {
                ALPAKA_FN_ACC_NO_CUDA static auto remainder(
                    RemainderStl const & remainder,
                    Tx const & x,
                    Ty const & y)
                -> decltype(std::remainder(x, y))
                {
                    boost::ignore_unused(remainder);
                    return std::remainder(x, y);
                }
            };
        }
    }
}
