/* Copyright 2016-2021 Marco Garten, Axel Huebl
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

/** @file AlgorithmThomasFermi.hpp
 *
 * IONIZATION ALGORITHM for the Thomas-Fermi model
 * - implements the calculation of the new number of free macro electrons
 *   from the Thomas-Fermi average charge state
 * - is called with the IONIZATION MODEL, specifically by setting the flag in
 *   @see speciesDefinition.param
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
#include "picongpu/particles/traits/GetAtomicNumbers.hpp"
#include "picongpu/traits/attribute/GetChargeState.hpp"

#include <pmacc/algorithms/math/floatMath/floatingPoint.tpp>

namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            /** AlgorithmThomasFermi
             *
             * ionization prediction for the Thomas-Fermi ionization model
             *
             */
            struct AlgorithmThomasFermi
            {
                /** Detailed Balance implementation of the Thomas-Fermi model
                 *
                 * This model uses local ion density and "temperature" values as input
                 * parameters to calculate an average charge state.
                 * A physical temperature requires a defined equilibrium state.
                 * Typical high power laser-plasma interaction is highly
                 * non-equilibrated, though. The name "temperature" is kept to illustrate
                 * the origination from the Thomas-Fermi model. It is nevertheless
                 * more accurate to think of it as an averaged kinetic energy
                 * which is not backed by the model and should therefore only be used with
                 * a certain suspicion in such Non-LTE scenarios.
                 *
                 * @tparam ParticleType type of particle for which to calculate
                 *     an average charge state
                 *
                 * @param temperature electron "temperature" value calculated from average
                 *        kinetic electron energy per ion in units of eV
                 * @param massDensity ion mass density in units of g/cm^3
                 *
                 * @return average charge state prediction according to the Thomas-Fermi model
                 */
                template<typename ParticleType>
                HDINLINE float_X detailedBalanceThomasFermi(
                    float_X const temperature,
                    float_X const massDensity,
                    ParticleType& parentIon)
                {
                    /* @TODO replace the float_64 with float_X and make sure the values are scaled to PIConGPU units */
                    constexpr float_64 protonNumber = GetAtomicNumbers<ParticleType>::type::numberOfProtons;
                    constexpr float_64 neutronNumber = GetAtomicNumbers<ParticleType>::type::numberOfNeutrons;

                    /* atomic mass number (usually A) A = N + Z */
                    constexpr float_64 massNumber = neutronNumber + protonNumber;

                    float_64 const T_0 = temperature / math::pow(protonNumber, float_64(4. / 3.));

                    float_64 const T_F = T_0 / (float_64(1.) + T_0);

                    /* for all the fitting parameters @see ionizer.param */

                    /** this is weird - I have to define temporary variables because
                     * otherwise the math::pow function won't recognize those at the
                     * exponent position */
                    constexpr float_64 TFA2_temp = thomasFermi::TFA2;
                    constexpr float_64 TFA4_temp = thomasFermi::TFA4;
                    constexpr float_64 TFBeta_temp = thomasFermi::TFBeta;

                    float_64 const A = thomasFermi::TFA1 * math::pow(T_0, TFA2_temp)
                        + thomasFermi::TFA3 * math::pow(T_0, TFA4_temp);

                    float_64 const B = -math::exp(
                        thomasFermi::TFB0 + thomasFermi::TFB1 * T_F
                        + thomasFermi::TFB2 * math::pow(T_F, float_64(7.)));

                    float_64 const C = thomasFermi::TFC1 * T_F + thomasFermi::TFC2;

                    constexpr float_64 invAtomicTimesMassNumber = float_64(1.) / (protonNumber * massNumber);
                    float_64 const R = massDensity * invAtomicTimesMassNumber;

                    float_64 const Q_1 = A * math::pow(R, B);

                    float_64 const Q = math::pow(math::pow(R, C) + math::pow(Q_1, C), float_64(1.) / C);

                    float_64 const x = thomasFermi::TFAlpha * math::pow(Q, TFBeta_temp);

                    /* Thomas-Fermi average ionization state */
                    float_X const ZStar = static_cast<float_X>(
                        protonNumber * x / (float_64(1.) + x + math::sqrt(float_64(1.) + float_64(2.) * x)));

                    return ZStar;
                }

                /** Functor implementation
                 *
                 * Calling this functor gives a prediction for an integer number of new
                 * free macro electrons to create. This prediction is based on the
                 * average charge state in the Thomas-Fermi model.
                 * The functor calculates the integer number of bound electrons from
                 * this state by a Monte-Carlo step.
                 *
                 * @tparam ParticleType type of particle to be ionized
                 *
                 * @param ZStar average charge state in the Thomas-Fermi model
                 * @param parentIon particle instance to be ionized
                 * @param randNr random number
                 *
                 * @return numNewFreeMacroElectrons number of new macro electrons to
                 *         create, range: [0, boundElectrons]
                 */
                template<typename ParticleType>
                HDINLINE uint32_t operator()(
                    float_X const kinEnergyDensity,
                    float_X const density,
                    ParticleType& parentIon,
                    float_X randNr)
                {
                    /* initialize functor return value: number of new macro electrons to create */
                    uint32_t numNewFreeMacroElectrons = 0u;

                    float_64 const densityUnit
                        = static_cast<float_64>(particleToGrid::derivedAttributes::Density().getUnit()[0]);
                    float_64 const kinEnergyDensityUnit
                        = static_cast<float_64>(particleToGrid::derivedAttributes::EnergyDensity().getUnit()[0]);
                    /* convert from kinetic energy density to average kinetic energy per particle */
                    float_64 const kinEnergyUnit = kinEnergyDensityUnit / densityUnit;
                    float_64 const avgKinEnergy = kinEnergyDensity / density * kinEnergyUnit;
                    /** convert kinetic energy in J to "temperature" in eV by assuming an ideal electron gas
                     * E_kin = 3/2 k*T
                     */
                    constexpr float_64 convKinEnergyToTemperature
                        = UNITCONV_Joule_to_keV * float_64(1.e3) * float_64(2. / 3.);
                    /** electron "temperature" in electron volts */
                    float_64 const temperature = avgKinEnergy * convKinEnergyToTemperature;

                    /* conversion factors from number density to mass density */
                    constexpr float_64 nAvogadro = SI::N_AVOGADRO;
                    constexpr float_64 convM3ToCM3 = 1.e6;

                    /* @TODO replace the float_64 with float_X and make sure the values are scaled to PIConGPU units */
                    constexpr float_64 protonNumber = GetAtomicNumbers<ParticleType>::type::numberOfProtons;
                    constexpr float_64 neutronNumber = GetAtomicNumbers<ParticleType>::type::numberOfNeutrons;

                    /* atomic mass number (usually A) A = N + Z */
                    constexpr float_64 massNumber = neutronNumber + protonNumber;

                    float_64 const convToMassDensity = densityUnit * massNumber / nAvogadro / convM3ToCM3;
                    /** mass density in units of g/cm^3 */
                    float_64 const massDensity = density * convToMassDensity;

                    /** lower ion density cutoff
                     *
                     * The Thomas-Fermi model yields unphysical artifacts for low densities.
                     * If `density` is lower than a user-definable ion number density value the model will not be
                     * applied.
                     */
                    constexpr float_X lowerDensityCutoff = particles::ionization::thomasFermi::CUTOFF_LOW_DENSITY;
                    /** lower electron temperature cutoff
                     *
                     * The Thomas-Fermi model also yields partly unphysical artifacts for low electron temperatures.
                     * If `temperature` is lower than a user-definable ion number temperature value the model will not
                     * be applied.
                     */
                    constexpr float_X lowerTemperatureCutoff
                        = particles::ionization::thomasFermi::CUTOFF_LOW_TEMPERATURE_EV;

                    if(density * densityUnit >= lowerDensityCutoff && temperature >= lowerTemperatureCutoff)
                    {
                        float_64 const chargeState = attribute::getChargeState(parentIon);
                        /* @TODO replace the float_64 with float_X and make sure the values are scaled to PIConGPU
                         * units */
                        constexpr float_64 protonNumber = GetAtomicNumbers<ParticleType>::type::numberOfProtons;

                        /* only ionize not-fully ionized ions */
                        if(chargeState < protonNumber)
                        {
                            /* Thomas-Fermi calculation step:
                             * Determines the new average charge state for each ion under
                             * LTE conditions.
                             */
                            float_X const ZStar = detailedBalanceThomasFermi(temperature, massDensity, parentIon);

                            /* integral part of the average charge state */
                            float_X intZStar;
                            /* fractional part of the average charge state */
                            float_X const fracZStar = pmacc::math::modf(ZStar, &intZStar);

                            /* Determine new charge state.
                             * We do a Monte-Carlo step to distribute charge states between
                             * the two "surrounding" integer numbers if ZStar has a non-zero
                             * fractional part.
                             */
                            float_X const newChargeState = intZStar + float_X(1.0) * (randNr < fracZStar);

                            /* define number of bound macro electrons before ionization */
                            float_X const prevBoundElectrons = parentIon[boundElectrons_];

                            /** determine the new number of bound electrons from the TF ionization state
                             * @TODO introduce partial macroparticle ionization / ionization distribution at some point
                             */
                            float_X const newBoundElectrons = protonNumber - newChargeState;

                            /* Only account for ionization: we only increase the charge
                             * state of an ion if necessary, but ignore recombination of
                             * electrons as prediced by the implemented detailed balance
                             * algorithm.
                             */
                            if(prevBoundElectrons > newBoundElectrons)
                                /* determine number of new free macro electrons
                                 * to be created in the ionization routine
                                 */
                                numNewFreeMacroElectrons
                                    = static_cast<uint32_t>(prevBoundElectrons - newBoundElectrons);
                        }
                    }

                    return numNewFreeMacroElectrons;
                }
            };

        } // namespace ionization
    } // namespace particles
} // namespace picongpu
