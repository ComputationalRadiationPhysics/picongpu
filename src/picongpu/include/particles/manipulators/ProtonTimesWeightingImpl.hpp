/**
 * Copyright 2015-2016 Axel Huebl
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

#include "particles/manipulators/ProtonTimesWeightingImpl.def"
#include "particles/traits/GetAtomicNumbers.hpp"

#include "simulation_defines.hpp"

namespace picongpu
{
namespace particles
{
namespace manipulators
{

struct ProtonTimesWeightingImpl
{

    template<typename T_SpeciesType>
    struct apply
    {
        typedef ProtonTimesWeightingImpl type;
    };

    HINLINE ProtonTimesWeightingImpl(uint32_t)
    {
    }

    /* Increase weighting of particleDest by proton number of SrcParticle
     *
     * The frame's `atomicNumber` `numberOfProtons`of `T_SrcParticle`
     * is used to increase the weighting of particleDest.
     * Useful to increase the weighting of macro electrons when cloned from an
     * ion with Z>1. Otherwise one would need Z macro electrons (each with the
     * same weighting as the initial ion) to keep the charge of a pre-ionized
     * atom neutral.
     *
     * \tparam T_DestParticle type of the particle species with weighting to manipulate
     * \tparam T_SrcParticle type of the particle species with proton number Z
     *
     * \see picongpu::particles::ManipulateDeriveSpecies , picongpu::kernelCloneParticles
     */
    template<typename T_DestParticle, typename T_SrcParticle>
    DINLINE void operator()(const DataSpace<simDim>&,
                            T_DestParticle& particleDest, T_SrcParticle&,
                            const bool isDestParticle, const bool isSrcParticle)
    {
        if (isDestParticle && isSrcParticle)
        {
            const float_X protonNumber = traits::GetAtomicNumbers<T_SrcParticle>::type::numberOfProtons;
            particleDest[weighting_] *= protonNumber;
        }
    }
};

} //namespace manipulators
} //namespace particles
} //namespace picongpu
