/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera, Heiko Burau, Marco Garten
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/algorithms/KinEnergy.hpp"
#include "picongpu/defines.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/EnergyDensityCutoff.def"
#include "picongpu/particles/particleToGrid/derivedAttributes/IsWeighted.hpp"
#include "picongpu/traits/attribute/GetMass.hpp"

#include <type_traits>

namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace derivedAttributes
            {
                template<class T_ParamClass>
                template<class T_Particle>
                DINLINE float_X EnergyDensityCutoff<T_ParamClass>::operator()(T_Particle& particle) const
                {
                    using ParamClass = T_ParamClass;

                    /* read existing attributes */
                    float_X const weighting = particle[weighting_];
                    float3_X const mom = particle[momentum_];
                    float_X const mass = picongpu::traits::attribute::getMass(weighting, particle);

                    constexpr float_X invCellVolume = float_X(1.0) / sim.pic.getCellSize().productOfComponents();

                    /* value for energy cut-off */
                    float_X const cutoffMaxEnergy = ParamClass::cutoffMaxEnergy;
                    float_X const cutoff = cutoffMaxEnergy / sim.unit.energy() * weighting;

                    float_X const kinEnergy = KinEnergy<>()(mom, mass);

                    float_X result(0.);
                    if(kinEnergy < cutoff)
                        result = kinEnergy * invCellVolume;

                    return result;
                }

                /** Energy density cutoff is weighted
                 *
                 * @tparam T_ParamClass parameter class containing the maximum energy cutoff
                 */
                template<typename T_ParamClass>
                struct IsWeighted<EnergyDensityCutoff<T_ParamClass>> : std::true_type
                {
                };
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
