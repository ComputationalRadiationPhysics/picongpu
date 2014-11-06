/**
 * Copyright 2014 Rene Widera, Marco Garten
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

namespace picongpu
{

template<typename T_Frame>
HDINLINE float_X getCharge(float_X weighting);

template<typename T_Frame>
HDINLINE float_X getCharge(float_X weighting, int chargeState);

/* For electrons it would be okay to do that framewise but not for ions
 * because they can differ in their respective charge state. */
template<typename T_Frame>
HDINLINE float_X getCharge(float_X weighting,const T_Frame&, int chargeState)
{
    return getCharge<T_Frame>(weighting, chargeState);
}

}// namespace picongpu
