/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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
#include "particles/manipulators/IManipulator.def"

namespace picongpu
{

namespace particles
{
namespace manipulators
{

template<typename T_Base>
struct IManipulator : private T_Base
{
    typedef T_Base Base;

    HINLINE IManipulator(uint32_t currentStep) : Base(currentStep)
    {
    }

    template<typename T_Particle1, typename T_Particle2>
    DINLINE void operator()(const DataSpace<simDim>& localCellIdx,
                            T_Particle1& particleSpecies1, T_Particle2& particleSpecies2,
                            const bool isParticle1, const bool isParticle2)
    {
        return Base::operator()(localCellIdx, particleSpecies1, particleSpecies2, isParticle1, isParticle2);
    }
};

} //namespace manipulators
} //namespace particles
} //namespace picongpu
