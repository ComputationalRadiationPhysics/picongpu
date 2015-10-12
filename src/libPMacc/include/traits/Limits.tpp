/**
 * Copyright 2014 Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "traits/Limits.hpp"

#include <climits>

namespace PMacc
{
namespace traits
{
namespace limits
{

template<>
struct Max<int>
{
    BOOST_STATIC_CONSTEXPR int value=INT_MAX;
};

template<>
struct Max<uint32_t>
{
    BOOST_STATIC_CONSTEXPR uint32_t value=static_cast<uint32_t>(-1);
};

template<>
struct Max<uint64_t>
{
    BOOST_STATIC_CONSTEXPR uint64_t value=static_cast<uint64_t>(-1);
};

} //namespace limits
} //namespace traits
} //namespace PMacc
