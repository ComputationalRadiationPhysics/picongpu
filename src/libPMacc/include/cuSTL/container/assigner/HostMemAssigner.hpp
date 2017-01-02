/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera
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

#include "cuSTL/algorithm/host/Foreach.hpp"
#include "lambda/placeholder.h"
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/int.hpp>
#include <stdint.h>

namespace PMacc
{
namespace assigner
{

namespace bmpl = boost::mpl;

template<typename T_Dim = bmpl::_1, typename T_CartBuffer = bmpl::_2>
struct HostMemAssigner
{
    static constexpr int dim = T_Dim::value;
    typedef T_CartBuffer CartBuffer;

    template<typename Type>
    HINLINE void assign(const Type& value)
    {
        // "Curiously recurring template pattern"
        CartBuffer* buffer = static_cast<CartBuffer*>(this);

        using namespace lambda;
        algorithm::host::Foreach foreach;
        foreach(buffer->zone(), buffer->origin(), _1 = value);
    }
};

} // assigner
} // PMacc

