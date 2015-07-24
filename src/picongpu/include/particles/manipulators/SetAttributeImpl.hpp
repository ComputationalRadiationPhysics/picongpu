/**
 * Copyright 2015 Marco Garten
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

template<typename T_ParamClass, typename T_ValueFunctor, typename T_SpeciesType>
struct SetAttributeImpl : private T_ValueFunctor
{
    typedef T_ParamClass ParamClass;
    typedef T_SpeciesType SpeciesType;
    typedef typename MakeIdentifier<SpeciesType>::type SpeciesName;

    typedef T_ValueFunctor ValueFunctor;

    HINLINE SetAttributeImpl(uint32_t currentStep)
    {

    }

    template<typename T_Particle1, typename T_Particle2>
    DINLINE void operator()(const DataSpace<simDim>& localCellIdx,
                            T_Particle1& particle, T_Particle2&,
                            const bool isParticle, const bool)
    {
        typedef T_Particle1 Particle;
        typedef typename SpeciesType::FrameType FrameType;

        if (isParticle)
        {
            /** Number of bound electrons at initial state of the neutral atom */
            const float_X protonNumber = GetAtomicNumbers<Particle>::type::numberOfProtons;

            /* in this case: 'assign' the number of protons to the number of bound electrons
             * \see particleConfig.param for the ValueFunctor */
            ValueFunctor::operator()(particle[boundElectrons_], protonNumber);
        }
    }

};

} //namespace manipulators
} //namespace particles
} //namespace picongpu
