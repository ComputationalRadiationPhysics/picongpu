/* Copyright 2013-2021 Heiko Burau, Rene Widera, Richard Pausch, Finn-Ole Carstens
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

#include "Particle.hpp"


namespace picongpu
{
    namespace plugins
    {
        namespace transitionRadiation
        {
            using complex_X = pmacc::math::Complex<float_X>;
            using complex_64 = pmacc::math::Complex<float_64>;

            /* Arbitrary margin which is necessary to prevent division by 0 error
             * created by particles moving in the plane of the foil.
             */
            float_X const DIV_BY_ZERO_MINIMUM = 1.e-7;

            /** Calculator class for calculation of transition radiation.
             *
             * @param particleSet transitionRadiation::Particle to compute transition radiation for
             * @param lookDirection vector of observation direction
             */
            class Calculator
            {
            private:
                transitionRadiation::Particle const& particle;
                float3_X const& lookDirection;

                float_X parMomSinTheta;
                float_X parMomCosTheta;
                float_X const parMomPhi;
                float_X parMomSinPhi;
                float_X parMomCosPhi;
                float_X detectorSinTheta;
                float_X detectorCosTheta;
                float_X const detectorPhi;
                float_X const uSquared;
                float_X const parSqrtOnePlusUSquared;

            public:
                HDINLINE
                Calculator(transitionRadiation::Particle const& particleSet, float3_X const& lookDirection)
                    : particle(particleSet)
                    , lookDirection(lookDirection)
                    , parMomPhi(particle.getMomPhi())
                    ,
                    // one has to add pi to the polar angle, because phi is in the range of 0 to 2 \pi
                    detectorPhi(picongpu::math::atan2(lookDirection.z(), lookDirection.x()) + picongpu::PI)
                    , uSquared(particle.getU() * particle.getU())
                    , parSqrtOnePlusUSquared(picongpu::math::sqrt(1 + uSquared))
                {
                    // frequent calculations
                    // momentum Space for Particle:
                    pmacc::math::sincos(particle.getMomTheta(), parMomSinTheta, parMomCosTheta);
                    pmacc::math::sincos(parMomPhi - detectorPhi, parMomSinPhi, parMomCosPhi);

                    // detector Position since lookDirection is normalized
                    float_X const detectorTheta = picongpu::math::acos(lookDirection.y());

                    pmacc::math::sincos(detectorTheta, detectorSinTheta, detectorCosTheta);
                }

                /** Perpendicular part of normalized energy
                 *
                 * Calculates perpendicular part to movement direction of normalized energy
                 * determined by formula:
                 * @f[E_{perp} = (u^2 \cos{\psi} \sin{\psi} \sin{\phi} \cos{\theta}) /
                 *          ((\sqrt{1 + u^2} - u \sin{\psi} \cos{\phi} \sin{\theta})^2 - u^2 \cos{\phi}^2
                 * \cos{\theta}^2)@f] where \psi is the azimuth angle of the particle momentum and \theta is the
                 * azimuth angle of the detector position to the movement direction y
                 *
                 * @return perpendicular part of normalized energy
                 */
                HDINLINE
                float_X calcEnergyPerp() const
                {
                    // a, x and y are temporary variables without an explicit physical meaning
                    float_X const a = uSquared * parMomCosTheta * parMomSinTheta * parMomSinPhi * detectorCosTheta;

                    // Denominator
                    float_X const x
                        = parSqrtOnePlusUSquared - particle.getU() * parMomSinTheta * parMomCosPhi * detectorSinTheta;
                    float_X const y = particle.getU() * parMomCosTheta * detectorCosTheta;

                    float_X denominator = x * x - y * y;

                    // Preventing division by 0
                    if(math::abs(denominator) < DIV_BY_ZERO_MINIMUM)
                    {
                        if(denominator < 0.0)
                            denominator = -DIV_BY_ZERO_MINIMUM;
                        else
                            denominator = DIV_BY_ZERO_MINIMUM;
                    }

                    return a / denominator;
                }

                /** Parallel part of normalized energy
                 *
                 * Calculates parallel part to movement direction of normalized energy
                 * determined by formula:
                 * @f[E_{para} = (u \cos{\psi} (u \sin{\psi} \cos{\phi} - \sqrt{1 + u^2} \sin{\theta}) /
                 *          ((\sqrt{1 + u^2} - u \sin{\psi} \cos{\phi} \sin{\theta})^2 - u^2 \cos{\phi}^2
                 * \cos{\theta}^2)@f] where \psi is the azimuth angle of the particle momentum and \theta is the
                 * azimuth angle of the detector position to the movement direction y
                 *
                 * @return parallel part of normalized energy
                 */
                HDINLINE
                float_X calcEnergyPara() const
                {
                    // a, b, c, x and y are just temporary variables without an explicit physical meaning
                    float_X const a = particle.getU() * parMomCosTheta;
                    float_X const b = particle.getU() * parMomSinTheta * parMomCosPhi;
                    float_X const c = parSqrtOnePlusUSquared * detectorSinTheta;

                    // Denominator
                    float_X const x
                        = parSqrtOnePlusUSquared - particle.getU() * parMomSinTheta * parMomCosPhi * detectorSinTheta;
                    float_X const y = particle.getU() * parMomCosTheta * detectorCosTheta;

                    float_X denominator = x * x - y * y;

                    // Preventing division by 0
                    if(math::abs(denominator) < DIV_BY_ZERO_MINIMUM)
                    {
                        if(denominator < 0.0)
                            denominator = -DIV_BY_ZERO_MINIMUM;
                        else
                            denominator = DIV_BY_ZERO_MINIMUM;
                    }

                    return a * (b - c) / denominator;
                }

                /** Exponent of form factor
                 *
                 * Calculates the exponent of the formfactor divided by \omega
                 * It represents the phase of a single electron in the bunch, but it is mostly
                 * calculated for performance reasons.
                 * \f[ F_exp = - i z ( 1 / v - \sin{\theta} \sin{\psi} \cos{\phi_P - \phi_D} / c ) / \cos{\phi}
                 *          - i \sin{\theta} \rho \cos{\phi_P - \phi_D} \f]
                 *
                 */
                HDINLINE
                complex_X calcFormFactorExponent() const
                {
                    // If case for longitudinal moving particles... leads to 0 later in the kernel
                    if(math::abs(parMomCosTheta) <= DIV_BY_ZERO_MINIMUM)
                        return complex_X(-1.0, 0.0);

                    float_X const a = detectorSinTheta * parMomSinTheta * math::cos(parMomPhi - detectorPhi);
                    float_X const b
                        = -(particle.getPosPara()) * (1 / particle.getVel() - a / SPEED_OF_LIGHT) / (parMomCosTheta);
                    float_X const c
                        = -detectorSinTheta * particle.getPosPerp() * math::cos(particle.getPosPhi() - detectorPhi);

                    complex_X const fpara = complex_X(0.0, b);
                    complex_X const fperp = complex_X(0.0, c);
                    return fpara + fperp;
                }
            }; // class Calculator

            /** Formfactor
             *
             * Calculates of the electron bunch with the exponent calculated by the
             * Calculator class.
             *
             * @f[F = \exp{ F_{exp} * \omega }@f]
             *
             * @param omega observed frequency
             * @param exponent exponent of exponential function
             */
            HDINLINE
            complex_X calcFormFactor(float_X const omega, complex_X const exponent)
            {
                // preventing division by 0
                const bool longMovingParticle = exponent.get_real() == -1.0;
                return float_X(longMovingParticle) * complex_X(0.0, 0.0)
                    + float_X(!longMovingParticle) * complex_X(math::exp(exponent * omega));
            }

        } // namespace transitionRadiation
    } // namespace plugins
} // namespace picongpu
