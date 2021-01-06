/* Copyright 2016-2021 Marco Garten, Jakob Trojok
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

#include <pmacc/types.hpp>
#include "picongpu/simulation_defines.hpp"
#include "picongpu/particles/traits/GetAtomicNumbers.hpp"
#include "picongpu/particles/traits/GetIonizationEnergies.hpp"
#include "picongpu/traits/attribute/GetChargeState.hpp"
#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/algorithms/math/floatMath/floatingPoint.tpp>
#include "picongpu/particles/ionization/utilities.hpp"
#include "picongpu/particles/ionization/byField/IonizationCurrent/IonizerReturn.hpp"

/** @file AlgorithmKeldysh.hpp
 *
 * - implements the calculation of ionization probability and returns the number of free electrons
 * - is called with the IONIZATION MODEL, specifically by setting the flag in @see speciesDefinition.param */


namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            /** Calculation for the Keldysh ionization model
             *
             * for linear laser polarization
             */
            struct AlgorithmKeldysh
            {
                /** Functor implementation
                 * \tparam EType type of electric field
                 * \tparam BType type of magnetic field
                 * \tparam ParticleType type of particle to be ionized
                 *
                 * \param bField magnetic field value at t=0
                 * \param eField electric field value at t=0
                 * \param parentIon particle instance to be ionized with position at t=0 and momentum at t=-1/2
                 * \param randNr random number, equally distributed in range [0.:1.0]
                 *
                 * \return ionization energy and number of new macro electrons to be created
                 */
                template<typename EType, typename BType, typename ParticleType>
                HDINLINE IonizerReturn
                operator()(const BType bField, const EType eField, ParticleType& parentIon, float_X randNr)
                {
                    const float_X protonNumber = GetAtomicNumbers<ParticleType>::type::numberOfProtons;
                    float_X chargeState = attribute::getChargeState(parentIon);

                    /* verify that ion is not completely ionized */
                    if(chargeState < protonNumber)
                    {
                        uint32_t const cs = pmacc::math::float2int_rd(chargeState);
                        const float_X iEnergy = typename GetIonizationEnergies<ParticleType>::type{}[cs];

                        constexpr float_X pi = pmacc::math::Pi<float_X>::value;
                        /* electric field in atomic units - only absolute value */
                        float_X eInAU = math::abs(eField) / ATOMIC_UNIT_EFIELD;

                        /* factor two avoid calculation math::pow(2,5./4.); */
                        const float_X twoToFiveQuarters = 2.3784142300054;

                        /* characteristic exponential function argument */
                        const float_X charExpArg = math::sqrt(util::cube(float_X(2.) * iEnergy)) / eInAU;

                        /* ionization rate */
                        float_X rateKeldysh = math::sqrt(float_X(6.) * pi) / twoToFiveQuarters * iEnergy
                            * math::sqrt(float_X(1.) / charExpArg) * math::exp(-float_X(2. / 3.) * charExpArg);

                        /* simulation time step in atomic units */
                        const float_X timeStepAU = float_X(DELTA_T / ATOMIC_UNIT_TIME);
                        /* ionization probability
                         *
                         * probability = rate * time step
                         * --> for infinitesimal time steps
                         *
                         * the whole ensemble should then follow
                         * P = 1 - exp(-rate * time step) if the laser wavelength is
                         * sampled well enough
                         */
                        float_X const probKeldysh = rateKeldysh * timeStepAU;

                        /* ionization condition */
                        if(randNr < probKeldysh)
                        {
                            /* return ionization energy number of macro electrons to produce */
                            return IonizerReturn{iEnergy, 1u};
                        }
                    }
                    /* no ionization */
                    return IonizerReturn{0.0, 0u};
                }
            };

        } // namespace ionization
    } // namespace particles
} // namespace picongpu
