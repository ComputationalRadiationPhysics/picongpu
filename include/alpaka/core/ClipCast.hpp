/**
* \file
* Copyright 2018 Benjamin Worpitz
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

#include <alpaka/meta/Integral.hpp>

#include <algorithm>
#include <limits>

namespace alpaka
{
    namespace core
    {
        //-----------------------------------------------------------------------------
        //! \return The input casted and clipped to T.
        template<
            typename T,
            typename V>
        auto clipCast(
            V const & val)
        -> T
        {
            static_assert(std::is_integral<T>::value && std::is_integral<V>::value, "clipCast can not be called with non-integral types!");

            auto constexpr max = static_cast<V>(std::numeric_limits<alpaka::meta::LowerMax<T, V>>::max());
            auto constexpr min = static_cast<V>(std::numeric_limits<alpaka::meta::HigherMin<T, V>>::min());

            return static_cast<T>(std::max(min, std::min(max, val)));
        }
    }
}
