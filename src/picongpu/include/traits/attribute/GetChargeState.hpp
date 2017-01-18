/**
 * Copyright 2014-2017 Marco Garten, Rene Widera
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
#include "traits/HasIdentifier.hpp"
#include "algorithms/TypeCast.hpp"
#include "particles/traits/GetAtomicNumbers.hpp"

namespace picongpu
{
namespace traits
{
namespace attribute
{
namespace detail
{

/** Calculate the charge state of an atom / ion
 *
 * use attribute `boundElectrons` to calculate the charge state
 */
template<bool T_HasBoundElectrons>
struct LoadChargeState
{
    /** Functor implementation
     *
     * \return chargeState = number of electrons in neutral atom - number of currently bound electrons
     */
    template<typename T_Particle>
    HDINLINE float_X operator()(const T_Particle& particle)
    {
        const float_X protonNumber = GetAtomicNumbers<T_Particle>::type::numberOfProtons;
        return protonNumber - particle[boundElectrons_];
    }
};

/**  Calculate charge state of an atom / ion
 *
 * This is the fallback implementation to throw an error if no `boundElectrons`
 * are available for a species.
 */
template<>
struct LoadChargeState<false>
{

    template<typename T_Particle>
    HDINLINE void operator()(const T_Particle& particle)
    {
        PMACC_CASSERT_MSG(This_species_has_only_one_charge_state,1==2);
    }
};
} // namespace detail

/** get the charge state of a macro particle
 *
 * This function trait considers the `boundElectrons` attribute if it is set.
 * Charge states do not add up and also the various particles in a macro particle
 * do NOT have different charge states where one would average over them.
 *
 * @param particle a reference to a particle
 * @return charge of the macro particle
 */
template<typename T_Particle>
HDINLINE float_X getChargeState(const T_Particle& particle)
{
    typedef T_Particle ParticleType;
    typedef typename PMacc::traits::HasIdentifier<ParticleType, boundElectrons>::type hasBoundElectrons;
    return detail::LoadChargeState<hasBoundElectrons::value >()(particle);
}

}// namespace attribute
}// namespace traits
}// namespace picongpu
