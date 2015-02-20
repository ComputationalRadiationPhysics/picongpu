/**
 * Copyright 2014 Rene Widera
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
namespace particles
{
namespace manipulators
{

template<typename T_ParamClass, typename T_Functor, typename T_SpeciesType>
struct IfRelativeGlobalPositionImpl : private T_Functor
{
    typedef T_ParamClass ParamClass;
     typedef T_SpeciesType SpeciesType;
    typedef typename MakeIdentifier<SpeciesType>::type SpeciesName;
    typedef T_Functor Functor;

    HINLINE IfRelativeGlobalPositionImpl(uint32_t currentStep) : Functor(currentStep)
    {
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        globalDomainSize = subGrid.getGlobalDomain().size;
        localDomainOffset = subGrid.getLocalDomain().offset;
    }

    template<typename T_Particle>
    DINLINE void operator()(const DataSpace<simDim>& localCellIdx, T_Particle& particle, const bool isParticle)
    {
        typedef typename SpeciesType::FrameType FrameType;


        DataSpace<simDim> myCellPosition = localCellIdx + localDomainOffset;

        float_X relativePosition = float_X(myCellPosition[ParamClass::dimension]) /
            float_X(globalDomainSize[ParamClass::dimension]);

        const bool inRange=(ParamClass::lowerBound <= relativePosition &&
            relativePosition < ParamClass::upperBound);
        const bool particleInRange = isParticle&& inRange;

        Functor::operator()(localCellIdx, particle, particleInRange);

    }

private:

    DataSpace<simDim> localDomainOffset;
    DataSpace<simDim> globalDomainSize;
};

} //namespace manipulators
} //namespace particles
} //namespace picongpu
