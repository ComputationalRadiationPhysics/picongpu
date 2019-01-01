/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/particles/densityProfiles/IProfile.def"


namespace picongpu
{
namespace densityProfiles
{

template<typename T_Base>
struct IProfile : private T_Base
{

    using Base = T_Base;

    HINLINE IProfile(uint32_t currentStep) : Base(currentStep)
    {
    }

    HDINLINE float_X operator()(const DataSpace<simDim>& totalCellOffset)
    {
        return Base::operator()(totalCellOffset);
    }
};
}
}
