/**
 * Copyright 2016-2017 Marco Garten
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

#include "pmacc_types.hpp"
#include "simulation_defines.hpp"
#include "particles/traits/GetAtomicNumbers.hpp"
#include "traits/attribute/GetChargeState.hpp"
#include "algorithms/math/floatMath/floatingPoint.tpp"
#include "particles/ionization/byCollision/ThomasFermi/TFFittingParameters.def"

/** \file AlgorithmThomasFermi.hpp
 *
 * IONIZATION ALGORITHM for the Thomas-Fermi model
 *
 * - implements the calculation of average ionization degree and changes charge states
 *   by decreasing the number of bound electrons
 * - is called with the IONIZATION MODEL, specifically by setting the flag in @see speciesDefinition.param
 */

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
     * parameters. To be able to speak of a "temperature" an equilibrium state
     * would be required. Typical high power laser-plasma interaction is highly
     * non-equilibrated, though. The name "temperature" is kept to illustrate
     * the origination from the Thomas-Fermi model. It is nevertheless
     * more accurate to think of it as an averaged kinetic energy.
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
        HDINLINE void
        operator()( const KinEnergyDensityType kinEnergyDensity, const DensityType density, ParticleType& parentIon, float_X randNr )
        {

            /* @TODO replace the float_64 with float_X and make sure the values are scaled to PIConGPU units */
            const float_64 protonNumber = GetAtomicNumbers<ParticleType>::type::numberOfProtons;
            const float_64 neutronNumber = GetAtomicNumbers<ParticleType>::type::numberOfNeutrons;
            float_64 chargeState = attribute::getChargeState(parentIon);

            /* ionization condition */
            if (chargeState < protonNumber)
            {
                /* atomic mass number (usually A) A = N + Z */
                float_64 massNumber = neutronNumber + protonNumber;

                /** @TODO replace the static_cast<float_64> by casts to float_X
                 * or leave out entirely and compute everything in PIConGPU scaled units
                 */
                float_64 const densityUnit = static_cast<float_64>(particleToGrid::derivedAttributes::Density().getUnit()[0]);
                float_64 const kinEnergyDensityUnit = static_cast<float_64>(particleToGrid::derivedAttributes::EnergyDensity().getUnit()[0]);
                /* convert from kinetic energy density to average kinetic energy per particle */
                float_64 kinEnergy = (kinEnergyDensity / density) * (kinEnergyDensityUnit / densityUnit);
                /** convert kinetic energy in J to "temperature" in eV by assuming an ideal electron gas
                 * E_kin = 3/2 k*T
                 */
                float_64 temperature = kinEnergy * UNITCONV_Joule_to_keV * float_64(1.e3) * float_64(2./3.);

                float_64 T_0 = temperature/math::pow(protonNumber,float_64(4./3.));

                float_64 T_F = T_0 / (float_64(1.) + T_0);

                /* for all the fitting parameters @see TFFittingParameters.def */

                /** this is weird - I have to define temporary variables because
                 * otherwise the math::pow function won't recognize those at the
                 * exponent position */
                float_64 TFA2_temp = TFA2;
                float_64 TFA4_temp = TFA4;
                float_64 TFBeta_temp = TFBeta;

                float_64 A = TFA1 * math::pow(T_0,TFA2_temp) + TFA3 * math::pow(T_0,TFA4_temp);

                float_64 B = -math::exp(TFB0 + TFB1*T_F + TFB2*math::pow(T_F,float_64(7.)));

                float_64 C = TFC1 * T_F + TFC2;

                /** requires mass density in g/cm^3
                 * @TODO relocate constants to param file or leave out and calculate unitless
                 */
                float_64 const nAvogadro = 6.022e23;
                float_64 const convM3ToCM3 = 1.e6;

                float_64 massDensity = density * densityUnit * massNumber / nAvogadro / convM3ToCM3;
                float_64 R = massDensity/(protonNumber * massNumber);

                float_64 Q_1 = A * math::pow(R,B);

                float_64 Q = math::pow(math::pow(R,C) + math::pow(Q_1,C), float_64(1.)/C);

                float_64 x = TFAlpha * math::pow(Q,TFBeta_temp);

                /* Thomas-Fermi average ionization state */
                float_64 ZStar = protonNumber * x / (float_64(1.) + x + math::sqrt(float_64(1.) + float_64(2.)*x));

                /* integral part of the charge state */
                float_64 intZStar;
                /* fractional part of the charge state */
                float_X fracZStar = static_cast<float_X>(math::modf(ZStar,&intZStar));

                /* determine charge state */
                float_X chargeState = static_cast<float_X>(intZStar) + float_X(1.0)*(randNr < fracZStar);

                /** determine the new number of bound electrons from the TF ionization state
                 * @TODO introduce partial macroparticle ionization / ionization distribution at some point
                 */
                float_X newBoundElectrons = protonNumber - chargeState;
                /* safety check to avoid double counting since recombination is not yet implemented */
                if (newBoundElectrons < parentIon[boundElectrons_])
                    /* update the particle attribute only if more free electrons are to be created */
                    parentIon[boundElectrons_] = newBoundElectrons;
            }

        }
    };

} // namespace ionization
} // namespace particles
} // namespace picongpu
