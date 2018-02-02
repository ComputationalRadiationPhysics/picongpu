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

#include <alpaka/math/atan2/Traits.hpp>

#include <boost/core/ignore_unused.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library atan2.
        class Atan2Stl
        {
        public:
            using Atan2Base = Atan2Stl;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library atan2 trait specialization.
            template<
                typename Ty,
                typename Tx>
            struct Atan2<
                Atan2Stl,
                Ty,
                Tx,
                typename std::enable_if<
                    std::is_arithmetic<Ty>::value
                    && std::is_arithmetic<Tx>::value>::type>
            {
                ALPAKA_FN_ACC_NO_CUDA static auto atan2(
                    Atan2Stl const & abs,
                    Ty const & y,
                    Tx const & x)
                -> decltype(std::atan2(y, x))
                {
                    boost::ignore_unused(abs);
                    return std::atan2(y, x);
                }
            };
        }
    }
}
