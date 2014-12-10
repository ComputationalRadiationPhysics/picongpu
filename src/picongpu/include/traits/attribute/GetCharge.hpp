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
#include "traits/frame/GetCharge.hpp"
#include "traits/HasIdentifier.hpp"

namespace picongpu
{
namespace traits
{
namespace attribute
{
namespace detail
{

/** Calculate the real charge of a particle
 *
 * use attribute `chargeState` to calculate the charge
 */
template<bool T_HasChargeState>
struct LoadChargeState
{

    template<typename T_Particle>
    HDINLINE float_X operator()(const float_X partialResult, const T_Particle& particle)
    {
        return partialResult * particle[chargeState_];
    }
};

/**  Calculate the real charge of a particle
 *
 * This is the fallback implementation if no `chargeState` is available for a particle
 */
template<>
struct LoadChargeState<false>
{

    template<typename T_Particle>
    HDINLINE float_X operator()(const float_X partialResult, const T_Particle& particle)
    {
        return partialResult;
    }
};
} // namespace detail

/** get the charge of a makro particle
 *
 * This function trait take care to the `chargeState` attribute if it is set
 *
 * @param weighting weighting of the particle
 * @param particle a reference to a particle
 * @return charge of the makro particle
 */
template<typename T_Particle>
HDINLINE float_X getCharge(const float_X weighting, const T_Particle& particle)
{
    typedef T_Particle ParticleType;
    typedef typename PMacc::traits::HasIdentifier<ParticleType, chargeState>::type hasChargeState;
    return detail::LoadChargeState<hasChargeState::value >()(
                                                      traits::frame::getCharge<typename ParticleType::FrameType > () * weighting,
                                                      particle);
}

}// namespace attribute
}// namespace traits
}// namespace picongpu
