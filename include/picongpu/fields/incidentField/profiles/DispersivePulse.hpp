/*
 * SPDX-FileCopyrightText: Fabia Dietrich, Klaus Steiniger, Richard Pausch, Finn-Ole Carstens
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/incidentField/Functors.hpp"

#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/math/Complex.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace picongpu::fields::incidentField
{
    namespace profiles
    {
        namespace detail
        {
            /** Unitless DispersivePulse parameters
             *
             * @tparam T_Params user (SI) parameters
             */
            template<typename T_Params>
            struct DispersivePulseUnitless : public BaseParamUnitless<T_Params>
            {
                //! User SI parameters
                using Params = T_Params;

                //! Base unitless parameters
                using Base = BaseParamUnitless<T_Params>;

                // unit: none
                using Params::SPECTRAL_SUPPORT;

                // unit: sim.unit.length()
                static constexpr float_X W0 = static_cast<float_X>(Params::W0_SI / sim.unit.length());

                // rayleigh length in propagation direction
                static constexpr float_X rayleighLength
                    = pmacc::math::Pi<float_X>::value * W0 * W0 / Base::WAVE_LENGTH;

                // unit: sim.unit.time()
                // corresponds to period length of DFT
                static constexpr float_X INIT_TIME = static_cast<float_X>(Params::PULSE_INIT) * Base::PULSE_DURATION;

                // Dispersion parameters
                // unit: sim.unit.length() * sim.unit.time()
                static constexpr float_X SD
                    = static_cast<float_X>(Params::SD_SI / sim.unit.time() / sim.unit.length());
                // unit: rad * sim.unit.time()
                static constexpr float_X AD = static_cast<float_X>(Params::AD_SI / sim.unit.time());
                // unit: sim.unit.time()^2
                static constexpr float_X GDD
                    = static_cast<float_X>(Params::GDD_SI / sim.unit.time() / sim.unit.time());
                // unit: sim.unit.time()^3
                static constexpr float_X TOD
                    = static_cast<float_X>(Params::TOD_SI / sim.unit.time() / sim.unit.time() / sim.unit.time());
            };

            /** DispersivePulse incident E functor
             *
             * @tparam T_Params parameters
             */
            template<typename T_Params>
            struct DispersivePulseFunctorIncidentE
                : public DispersivePulseUnitless<T_Params>
                , public incidentField::detail::BaseFunctorE<T_Params>
            {
                //! Unitless parameters type
                using Unitless = DispersivePulseUnitless<T_Params>;

                //! Base functor type
                using Base = incidentField::detail::BaseFunctorE<T_Params>;

                /** Create a functor on the host side for the given time step
                 *
                 * @param currentStep current time step index, note that it is fractional
                 * @param unitField conversion factor from SI to internal units,
                 *                  fieldE_internal = fieldE_SI / unitField
                 */
                HINLINE DispersivePulseFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
                    : Base(currentStep, unitField)
                {
                }

                /** Calculate incident field E value for the given position
                 * The implementation is based on a definition of the electric field E given in frequency-space domain,
                 * which is printed in eq. (6) of ref. [1]. Compared to the reference, higher order derivatives of the
                 * frequency dependent propagation angle are set to zero, i.e. theta'' = theta''' = 0.
                 * Note, due to the different definitions of 'pulse duration' between PIConGPU and ref. [1], named
                 * `PULSE_DURATION` and `tau_0` respectively, it holds tau_0 = 2*PULSE_DURATION. Further, alpha, i.e.
                 * eq. (3) of [1], is implemented as expandedWaveVectorX().
                 *
                 * Dispersions are assumed to be along the polarization axis, i.e. Axis1 in laser internal coordinate
                 * system.
                 *
                 * For initialization in time-space domain, this field needs to be transformed from frequency to time
                 * domain, which is done via a discrete Fourier transform, see getValueE().
                 *
                 * References:
                 * [1] K. Steiniger, F. Dietrich et al., "Distortions in focusing laser pulses due to spatio-temporal
                 * couplings: an analytic description", High Power Laser Science and Engineering, vol. 12, p. e25,
                 * 2024. doi:10.1017/hpl.2023.96
                 *
                 * @param totalCellIdx cell index in the total domain (including all moving window slides)
                 * @return incident field E value in internal units
                 */
                HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
                {
                    if constexpr(Unitless::Polarisation == PolarisationType::Linear)
                        return this->getLinearPolarizationVector() * getValueE(totalCellIdx, 0.0_X);
                    else
                    {
                        auto const phaseShift = pmacc::math::Pi<float_X>::halfValue;
                        return this->getCircularPolarizationVector1() * getValueE(totalCellIdx, phaseShift)
                               + this->getCircularPolarizationVector2() * getValueE(totalCellIdx, 0.0_X);
                    }
                }

            private:
                /** Helper function to calculate the electric field in frequency domain.
                 * Initial frequency dependent complex phase expanded up to third order in (Omega - Omega_0).
                 * Takes only first order angular dispersion d theta / d Omega = theta^prime into account
                 * and neglects all higher order angular dispersion terms, e.g. theta^{prime prime},
                 * theta^{prime prime prime}, ...
                 *
                 * @param dOmega frequency difference Omega-Omega0 for which the E-value is calculated
                 */
                HDINLINE float_X expandedWaveVectorX(float_X const dOmega) const
                {
                    return Unitless::W0 / sim.pic.getSpeedOfLight()
                           * (Unitless::OMEGA0 * Unitless::AD * dOmega + Unitless::AD * dOmega * dOmega
                              - Unitless::OMEGA0 / 6.0_X * Unitless::AD * Unitless::AD * Unitless::AD * dOmega * dOmega
                                    * dOmega);
                }

                /** The following two functions provide the electric field in frequency domain
                 * E(Omega) = amp * exp(-i*phi)
                 * Please ensure that E(Omega = 0) = 0 (no constant field contribution), i.e. the pulse
                 * length has to be big enough. Otherwise, the implemented DFT will produce wrong results.
                 *
                 * @param pos position in laser internal coordinates at which the E-value is calculated
                 * @param Omega frequency for which the E-value is calculated
                 */
                HDINLINE float_X amp(float3_X const& pos, float_X const Omega) const
                {
                    float_X const dOmega = Omega - Unitless::OMEGA0;

                    // beam waist at the generation plane so that at focus we will get W0
                    float_X const waist
                        = Unitless::W0
                          * math::sqrt(
                              1.0_X + (pos[0] / Unitless::rayleighLength) * (pos[0] / Unitless::rayleighLength));

                    // Initial frequency dependent complex phase
                    float_X alpha = expandedWaveVectorX(dOmega);

                    // Center of a frequency's spatial distribution
                    float_X center = Unitless::SD * dOmega
                                     - sim.pic.getSpeedOfLight() * alpha * pos[0] / (Unitless::W0 * Unitless::OMEGA0);

                    // gaussian envelope in frequency domain
                    float_X const envFreqExp = -dOmega * dOmega * Unitless::PULSE_DURATION * Unitless::PULSE_DURATION;

                    /* Transverse envelopes
                     * Apply shift of frequency's distribution center only along polarization direction */
                    float_X const envYExp
                        = -(pos[1] - center) * (pos[1] - center) / (waist * waist); // envelope y - direction
                    float_X const envZExp = -pos[2] * pos[2] / (waist * waist); // envelope z - direction

                    float_X mag = math::exp(envFreqExp + envYExp + envZExp);

                    // distinguish between dimensions
                    if constexpr(simDim == DIM2)
                    {
                        // pos has just two entries: pos[0] as propagation direction and pos[1] as transversal
                        // direction
                        mag *= math::sqrt(Unitless::W0 / waist);
                    }
                    else if constexpr(simDim == DIM3)
                    {
                        mag *= Unitless::W0 / waist;
                    }

                    // Normalization to Amplitude
                    constexpr float_X pi = pmacc::math::Pi<float_X>::value;
                    mag *= math::sqrt(pi) * 2.0_X * Unitless::PULSE_DURATION * Unitless::AMPLITUDE;

                    return mag;
                }

                HDINLINE float_X phi(float3_X const& pos, float_X const Omega, float_X const phaseShift) const
                {
                    float_X const dOmega = Omega - Unitless::OMEGA0;

                    /* 'initial value' of phase due to distance of position where E-value is computed from focus
                     * and in-focus dispersion values */
                    float_X phase
                        = Omega * pos[0] / sim.pic.getSpeedOfLight() + 0.5_X * Unitless::GDD * dOmega * dOmega
                          + Unitless::TOD / 6.0_X * dOmega * dOmega * dOmega + phaseShift + Unitless::LASER_PHASE;

                    // Initial frequency dependent complex phase
                    float_X alpha = expandedWaveVectorX(dOmega);

                    // Center of a frequency's spatial distribution
                    float_X center = Unitless::SD * dOmega
                                     - sim.pic.getSpeedOfLight() * alpha * pos[0] / (Unitless::W0 * Unitless::OMEGA0);

                    // inverse radius of curvature of the pulse's wavefronts
                    auto const R_inv
                        = pos[0] / (Unitless::rayleighLength * Unitless::rayleighLength + pos[0] * pos[0]);

                    /* Parabolic phase curvature out of focus
                     * Apply shift of frequency's distribution center only along polarization direction */
                    phase += ((pos[1] - center) * (pos[1] - center) + pos[2] * pos[2]) * Omega * 0.5_X * R_inv
                             / sim.pic.getSpeedOfLight();

                    /* Off-axis part of propagation vector k due to dispersions
                     * Only applied along polarization direction */
                    phase
                        -= alpha * pos[1] / Unitless::W0 + 0.25_X * alpha * alpha * pos[0] / Unitless::rayleighLength;

                    // Apply Gouy phase shift
                    auto const xi = math::atan(pos[0] / Unitless::rayleighLength);
                    // distinguish between dimensions
                    if constexpr(simDim == DIM2)
                    {
                        phase -= 0.5_X * xi;
                    }
                    else if constexpr(simDim == DIM3)
                    {
                        phase -= xi;
                    }
                    return phase;
                }

                /** Get value of E field in time domain for the given position, using DFT
                 * Interpolation order of DFT given via timestep in simulation.param and INIT_TIME
                 * Neglecting the constant part of DFT (k = 0) because there should be no constant field
                 *
                 * @param totalCellIdx cell index in the total domain (including all moving window slides)
                 * @param phaseShift additional phase shift to add on top of everything else,
                 *                   in radian
                 */
                HDINLINE float_X getValueE(floatD_X const& totalCellIdx, float_X const phaseShift) const
                {
                    /* Check if current position is within the window of laser initialization.
                     *
                     * Only start laser initialization if the current position is within the window,
                     * as there would be periodic pulses fed into the simulation otherwise.
                     *
                     * The window of laser initialization is [-0.5*INIT_TIME, 0.5*INIT_TIME]*c along the central beam
                     * axis around the laser pulse maximum. Note, the position of the pulse maximum is influenced by
                     * TIME_DELAY.
                     *
                     * In case of oblique propagation of the pulse with respect to the Huygens surface, points on the
                     * Huygens surface can be within the window or outside the window depending on their transverse
                     * offset from the pulse's origin on the surface.
                     */
                    auto const shiftedTime = this->getTminusXoverC(totalCellIdx); // >0: inside, <0: outside

                    // Turn laser on the Huygens surface off, if position is outside the window of laser
                    // initialization.
                    if(shiftedTime < 0.0_X || shiftedTime > Unitless::INIT_TIME)
                        return 0.0_X;

                    // get simulation time step
                    auto const time = this->currentTimeOrigin;

                    /* Calculate position of laser origin on the incidentField plane relative to the focus.
                     * Coordinate system is PIConGPUs and not the laser internal coordinate system where X is
                     * the propagation direction.
                     */
                    float3_X const originRelativeToFocus = this->origin - this->focus;

                    /* Relative distance of the origin from the laser focus, transformed into the laser
                     * coordination system where X is the propagation direction. */
                    float_X const distanceOriginRelativeToFocus
                        = pmacc::math::dot(originRelativeToFocus, this->getAxis0());

                    /* Shifting pulse for half of INIT_TIME, TIME_DELAY, and travel time from the focus to
                     * the origin in order to start with the front of the laser pulse at the origin. */
                    constexpr auto mue = 0.5_X * Unitless::INIT_TIME + Unitless::TIME_DELAY;
                    float_X const timeDelay = mue - (distanceOriginRelativeToFocus / sim.pic.getSpeedOfLight());

                    // interpolation order of DFT
                    float_X N_raw = Unitless::INIT_TIME / sim.pic.getDt();
                    int const n = static_cast<int>(N_raw * 0.5_X); // -0 instead of -1 for rounding up N_raw

                    // frequency step for DFT
                    float_X const dOmk = pmacc::math::Pi<float_X>::doubleValue / Unitless::INIT_TIME;

                    // Since the (Gaussian) spectrum has only significant values near the central frequency,
                    // the summation over all frequencies is reduced to frequencies within an interval
                    // around the central frequency.

                    // standard deviation of the Gaussian distributed spectrum
                    float_X const sigma_Om = 1._X / (pmacc::math::sqrt(2._X) * Unitless::PULSE_DURATION);

                    // index of the mean frequency of the Gaussian distributed spectrum
                    // unit: [dOmk]
                    int const center_k = static_cast<int>(
                        sim.pic.getSpeedOfLight() * Unitless::INIT_TIME / Unitless::Base::WAVE_LENGTH);

                    // index of the lowest frequency in the Gaussian distributed spectrum which is used in the
                    // DFT 4*sigma_Om distance from central frequency
                    int const minOm_k = center_k - static_cast<int>(Unitless::SPECTRAL_SUPPORT * sigma_Om / dOmk);
                    int const k_min = math::max(minOm_k, 1);

                    // index of the highest frequency in the Gaussian distributed spectrum which is used in the
                    // DFT
                    int const maxOm_k = 2 * center_k - minOm_k;
                    int const k_max = math::min(maxOm_k, n);

                    // electric field in time-domain
                    float_X E_t = 0.0_X;

                    // transform current point of evaluation to laser internal coordinates
                    // The laser internal coordinate system's origin
                    // is in the focus and the laser propagates along the x-axis.
                    float3_X internalPosition = this->getInternalCoordinates(totalCellIdx);

                    // shifted time at which the field is evaluated
                    float_X evaluationTime = time - timeDelay;

                    /* E_t = sum(2 * amp(k) * (cos(phi) * cos(omegaT(k)) + sin(phi) * sin(omegaT(k)))) / dt) / (2n+1)
                     * == E_t = 2 * sum(amp(k) * (cos(phi) * cos(omegaT(k)) + sin(phi) * sin(omegaT(k))))) /
                     * (dt*(2n+1)) == E_t = 2 * sum(amp(k) * (cos(phi - omegaT(k)))) / (dt*(2n+1)) E_t = sum(amp(k) *
                     * (cos(phi - omegaT(k)))) * 2 / (dt*(2n+1))
                     */
                    for(int k = k_min; k <= k_max; k++)
                    {
                        // stores angular frequency for DFT-loop
                        float_X const Omk = static_cast<float_X>(k) * dOmk;
                        float_X const phiK = phi(internalPosition, Omk, phaseShift);
                        float_X const omegaTK = Omk * evaluationTime;
                        E_t += amp(internalPosition, Omk) * pmacc::math::cos(phiK - omegaTK);
                    }

                    /* Apply (standard) normalization of DFT, see above: E_t *= 2 / (dt*(2n+1))
                     * Additionally, divide amplitude E_t by 2 to compensate doubled spectral field strength
                     * introduced by assuming E(-Omega) = E(Omega) in the computation of the DFT,
                     * which also needs to be fulfilled for E(t) to be real.
                     */
                    E_t /= sim.pic.getDt() * static_cast<float_X>(2 * n + 1);

                    return E_t;
                } // getValueE
            }; // DispersivePulseFunctorIncidentE
        } // namespace detail

        template<typename T_Params>
        struct DispersivePulse
        {
            //! Get text name of the incident field profile
            HINLINE static std::string getName()
            {
                return "DispersivePulse";
            }
        };
    } // namespace profiles

    namespace traits::detail
    {
        /** Get type of incident field E functor for the dispersive laser profile type
         *
         * @tparam T_Params parameters
         */
        template<typename T_Params>
        struct GetFunctorIncidentE<profiles::DispersivePulse<T_Params>>
        {
            using type = profiles::detail::DispersivePulseFunctorIncidentE<T_Params>;
        };

        /** Get type of incident field B functor for the dispersive laser profile type
         *
         * @tparam T_Params parameters
         */
        template<typename T_Params>
        struct GetFunctorIncidentB<profiles::DispersivePulse<T_Params>>
        {
            using type = incidentField::detail::ApproximateIncidentB<
                typename GetFunctorIncidentE<profiles::DispersivePulse<T_Params>>::type>;
        };
    } // namespace traits::detail
} // namespace picongpu::fields::incidentField
