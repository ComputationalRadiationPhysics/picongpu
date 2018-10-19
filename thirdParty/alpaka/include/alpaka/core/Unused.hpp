/**
* \file
* Copyright 2018 Axel Huebl
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

#include <alpaka/core/Common.hpp>

#include <boost/config.hpp>

namespace alpaka
{
    ALPAKA_NO_HOST_ACC_WARNING
    template< typename... Ts >
    BOOST_FORCEINLINE
    BOOST_CXX14_CONSTEXPR
    ALPAKA_FN_HOST_ACC
    void
    ignore_unused( Ts const& ... )
    {}

    ALPAKA_NO_HOST_ACC_WARNING
    template< typename... Ts >
    BOOST_FORCEINLINE
    BOOST_CXX14_CONSTEXPR
    ALPAKA_FN_HOST_ACC
    void
    ignore_unused()
    {}

} // namespace alpaka

