/**
 * Copyright 2014-2016 Rene Widera
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
#include "particles/traits/GetAtomicNumbers.hpp"

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
 * use attribute `boundElectrons` and the proton number from
 * flag `atomicNumbers` to calculate the charge
 *
 * \tparam T_HasBoundElectrons boolean that describes if species allows multiple charge states
 * due to bound electrons
 */
template<bool T_HasBoundElectrons>
struct LoadBoundElectrons
{
    /** Functor implementation
     *
     * \tparam T_Particle particle type
     * \param singlyChargedResult charge resulting from multiplying a single
     * electron charge (positive OR negative) by the macro particle weighting
     * \param particle particle reference
     */
    template<typename T_Particle>
    HDINLINE float_X operator()(const float_X singlyChargedResult, const T_Particle& particle)
    {
        const float_X protonNumber = GetAtomicNumbers<T_Particle>::type::numberOfProtons;

        return singlyChargedResult * (protonNumber - particle[boundElectrons_]);
    }
};

/**  Calculate the real charge of a particle
 *
 * This is the fallback implementation if no `boundElectrons` are available for a particle
 *
 * \tparam T_HasBoundElectrons boolean that describes if species allows multiple charge states
 * due to bound electrons
 */
template<>
struct LoadBoundElectrons<false>
{
    /** Functor implementation
     *
     * \tparam T_Particle particle type
     * \param singlyChargedResult charge resulting from multiplying a single
     * electron charge (positive OR negative) by the macro particle weighting
     * \param particle particle reference
     */
    template<typename T_Particle>
    HDINLINE float_X operator()(const float_X singlyChargedResult, const T_Particle& particle)
    {
        return singlyChargedResult;
    }
};
} // namespace detail

/** get the charge of a macro particle
 *
 * This function trait considers the `boundElectrons` attribute if it is set
 *
 * @param weighting weighting of the particle
 * @param particle a reference to a particle
 * @return charge of the macro particle
 */
template<typename T_Particle>
HDINLINE float_X getCharge(const float_X weighting, const T_Particle& particle)
{
    typedef T_Particle ParticleType;
    typedef typename PMacc::traits::HasIdentifier<ParticleType, boundElectrons>::type hasBoundElectrons;
    return detail::LoadBoundElectrons<hasBoundElectrons::value >()(
                                                      frame::getCharge<typename ParticleType::FrameType > () * weighting,
                                                      particle);
}

}// namespace attribute
}// namespace traits
}// namespace picongpu
