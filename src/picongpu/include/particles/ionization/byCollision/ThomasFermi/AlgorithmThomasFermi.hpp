/* Copyright 2016-2017 Marco Garten
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

/** \file AlgorithmThomasFermi.hpp
 *
 * IONIZATION ALGORITHM for the Thomas-Fermi model
 * - implements the calculation of average ionization degree and returns the
 *   new number of bound electrons
 * - is called with the IONIZATION MODEL, specifically by setting the flag in
 *   @see speciesDefinition.param
 */

#pragma once

#include "simulation_defines.hpp"
#include "particles/traits/GetAtomicNumbers.hpp"
#include "traits/attribute/GetChargeState.hpp"

#include "algorithms/math/floatMath/floatingPoint.tpp"

namespace picongpu
{
namespace particles
{
namespace ionization
{

    /** \struct AlgorithmThomasFermi
     *
     * \brief calculation for the Thomas-Fermi pressure ionization model
     *
     * This model uses local density and "temperature" values as input
     * parameters. A physical temperature requires a defined equilibrium state.
     * Typical high power laser-plasma interaction is highly
     * non-equilibrated, though. The name "temperature" is kept to illustrate
     * the origination from the Thomas-Fermi model. It is nevertheless
     * more accurate to think of it as an averaged kinetic energy
     * which is not backed by the model and should therefore only be used with
     * a certain suspicion in such Non-LTE scenarios.
     */
    struct AlgorithmThomasFermi
    {
        /** Functor implementation
         *
         * \tparam DensityType type of number
         * \tparam KinEnergyDensityType type of kinetic energy density
         * \tparam ParticleType type of particle to be ionized
         *
         * \param density number density value
         * \param kinEnergyDensity kinetic energy density value
         * \param parentIon particle instance to be ionized
         * \param randNr random number
         */
        template<typename KinEnergyDensityType, typename DensityType, typename ParticleType >
        HDINLINE float_X
        operator()( const KinEnergyDensityType kinEnergyDensity, const DensityType density, ParticleType& parentIon, float_X randNr )
        {

            /* @TODO replace the float_64 with float_X and make sure the values are scaled to PIConGPU units */
            constexpr float_64 protonNumber = GetAtomicNumbers<ParticleType>::type::numberOfProtons;
            constexpr float_64 neutronNumber = GetAtomicNumbers<ParticleType>::type::numberOfNeutrons;
            float_64 chargeState = attribute::getChargeState(parentIon);

            /* ionization condition */
            if (chargeState < protonNumber)
            {
                /* atomic mass number (usually A) A = N + Z */
                constexpr float_64 massNumber = neutronNumber + protonNumber;

                /** @TODO replace the static_cast<float_64> by casts to float_X
                 * or leave out entirely and compute everything in PIConGPU scaled units
                 */
                float_64 const densityUnit = static_cast<float_64>(particleToGrid::derivedAttributes::Density().getUnit()[0]);
                float_64 const kinEnergyDensityUnit = static_cast<float_64>(particleToGrid::derivedAttributes::EnergyDensity().getUnit()[0]);
                /* convert from kinetic energy density to average kinetic energy per particle */
                float_64 const kinEnergyUnit = kinEnergyDensityUnit / densityUnit;
                float_64 const kinEnergy = (kinEnergyDensity / density) * kinEnergyUnit;
                /** convert kinetic energy in J to "temperature" in eV by assuming an ideal electron gas
                 * E_kin = 3/2 k*T
                 */
                constexpr float_64 convKinEnergyToTemperature = UNITCONV_Joule_to_keV * float_64(1.e3) * float_64(2./3.);
                float_64 const temperature = kinEnergy * convKinEnergyToTemperature;

                float_64 const T_0 = temperature/math::pow(protonNumber,float_64(4./3.));

                float_64 const T_F = T_0 / (float_64(1.) + T_0);

                /* for all the fitting parameters @see ionizerConfig.param */

                /** this is weird - I have to define temporary variables because
                 * otherwise the math::pow function won't recognize those at the
                 * exponent position */
                constexpr float_64 TFA2_temp = thomasFermi::TFA2;
                constexpr float_64 TFA4_temp = thomasFermi::TFA4;
                constexpr float_64 TFBeta_temp = thomasFermi::TFBeta;

                float_64 const A = thomasFermi::TFA1 * math::pow(T_0,TFA2_temp) + thomasFermi::TFA3 * math::pow(T_0,TFA4_temp);

                float_64 const B = -math::exp(thomasFermi::TFB0 + thomasFermi::TFB1*T_F + thomasFermi::TFB2*math::pow(T_F,float_64(7.)));

                float_64 const C = thomasFermi::TFC1 * T_F + thomasFermi::TFC2;

                /* requires mass density in g/cm^3 */
                constexpr float_64 nAvogadro = SI::N_AVOGADRO;
                constexpr float_64 convM3ToCM3 = 1.e6;

                float_64 const convToMassDensity = densityUnit * massNumber / nAvogadro / convM3ToCM3;
                float_64 const massDensity = density * convToMassDensity;

                constexpr float_64 invAtomicTimesMassNumber = float_64(1.) / (protonNumber * massNumber);
                float_64 const R = massDensity * invAtomicTimesMassNumber;

                float_64 const Q_1 = A * math::pow(R,B);

                float_64 const Q = math::pow(math::pow(R,C) + math::pow(Q_1,C), float_64(1.)/C);

                float_64 const x = thomasFermi::TFAlpha * math::pow(Q,TFBeta_temp);

                /* Thomas-Fermi average ionization state */
                float_64 const ZStar = protonNumber * x / (float_64(1.) + x + math::sqrt(float_64(1.) + float_64(2.)*x));

                /* integral part of the charge state */
                float_64 intZStar;
                /* fractional part of the charge state */
                float_X fracZStar = static_cast<float_X>(math::modf(ZStar,&intZStar));

                /* determine charge state */
                float_X const chargeState = static_cast<float_X>(intZStar) + float_X(1.0)*(randNr < fracZStar);

                /** determine the new number of bound electrons from the TF ionization state
                 * @TODO introduce partial macroparticle ionization / ionization distribution at some point
                 */
                float_X const newBoundElectrons = protonNumber - chargeState;

                return newBoundElectrons;
            }

            return float_X(0.);
        }
    };

} // namespace ionization
} // namespace particles
} // namespace picongpu
