/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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
#include "particles/startPosition/MacroParticleCfg.hpp"

namespace picongpu
{

namespace particles
{
namespace startPosition
{

template<typename T_Base>
struct IFunctor : private T_Base
{
    typedef T_Base Base;

    HINLINE IFunctor(uint32_t currentStep) : Base(currentStep)
    {
    }

    DINLINE void init(const DataSpace<simDim>& totalCellOffset)
    {
        Base::init(totalCellOffset);
    }

    DINLINE floatD_X operator()(const uint32_t currentParticleIdx)
    {
        return Base::operator()(currentParticleIdx);
    }

    DINLINE MacroParticleCfg mapRealToMacroParticle(const float_X realElPerCell)
    {
        return Base::mapRealToMacroParticle(realElPerCell);
    }
};

} //namespace startPosition
} //namespace particles
} //namespace picongpu
