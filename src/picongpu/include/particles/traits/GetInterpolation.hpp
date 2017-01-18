/**
 * Copyright 2014-2017 Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "simulation_defines.hpp"
#include "traits/GetFlagType.hpp"
#include "traits/Resolve.hpp"

namespace picongpu
{
namespace traits
{

template<typename T_Species>
struct GetInterpolation
{
    typedef typename PMacc::traits::Resolve<
        typename GetFlagType<typename T_Species::FrameType, interpolation<> >::type
    >::type type;
};
} //namespace traits

}// namespace picongpu
