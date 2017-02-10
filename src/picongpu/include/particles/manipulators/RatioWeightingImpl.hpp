/**
 * Copyright 2015-2017 Axel Huebl, Richard Pausch
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

#include "particles/manipulators/RatioWeightingImpl.def"
#include "particles/traits/GetDensityRatio.hpp"

#include "simulation_defines.hpp"

namespace picongpu
{
namespace particles
{
namespace manipulators
{

struct RatioWeightingImpl
{

    template<typename T_SpeciesType>
    struct apply
    {
        typedef RatioWeightingImpl type;
    };

    HINLINE RatioWeightingImpl(uint32_t)
    {
    }

    /* Adjust the weighting of particleDes by densityRatio of own & Src particle
     *
     * While deriving a particle (particleDes) from another (T_SrcParticle), one
     * can afterward directly normalize the weighting back to the intended density:
     * - divide weighting with the `T_SrcParticle`'s densityRatio
     *   (to get macro particle weighting according to reference GAS_DENSITY * profile
     *    at this specific point in space & time)
     * - multiply weighting with own densityRatio (to get this species'
     *    densityRatio * GAS_DENSITY * profile)
     *
     * This is useful when the profile and number of macro particles for both species
     * shall be the same and the initialization of another profile via `CreateGas`
     * would be expensive (or one wants to keep the exact same position while deriving).
     *
     * \tparam T_DesParticle type of the particle species with weighting to manipulate
     * \tparam T_SrcParticle type of the particle species one cloned from
     *
     * \see picongpu::particles::ManipulateDeriveSpecies , picongpu::kernelCloneParticles
     */
    template<typename T_DesParticle, typename T_SrcParticle>
    DINLINE void operator()(const DataSpace<simDim>&,
                            T_DesParticle& particleDes, T_SrcParticle&,
                            const bool isDesParticle, const bool isSrcParticle)
    {
        if (isDesParticle && isSrcParticle)
        {
            const float_X densityRatioDes =
                traits::GetDensityRatio<T_DesParticle>::type::getValue();
            const float_X densityRatioSrc =
                traits::GetDensityRatio<T_SrcParticle>::type::getValue();

            particleDes[weighting_] *= densityRatioDes / densityRatioSrc;
        }
    }
};

} //namespace manipulators
} //namespace particles
} //namespace picongpu
