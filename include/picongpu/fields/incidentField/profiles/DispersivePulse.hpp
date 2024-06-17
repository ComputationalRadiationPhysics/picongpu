/* Copyright 2022-2023 Fabia Dietrich, Klaus Steiniger, Richard Pausch, Finn-Ole Carstens
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/incidentField/Functors.hpp"
#include "picongpu/fields/incidentField/Traits.hpp"

#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/math/Complex.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>


namespace picongpu
{
    namespace fields
    {
        namespace incidentField
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

                        // unit: UNIT_LENGTH
                        static constexpr float_X W0 = static_cast<float_X>(Params::W0_SI / UNIT_LENGTH);

                        // rayleigh length in propagation direction
                        static constexpr float_X rayleighLength
                            = pmacc::math::Pi<float_X>::value * W0 * W0 / Base::WAVE_LENGTH;

                        // unit: UNIT_TIME
                        // corresponds to period length of DFT
                        static constexpr float_X INIT_TIME
                            = static_cast<float_X>(Params::PULSE_INIT) * Base::PULSE_DURATION;

                        // Dispersion parameters
                        // unit: UNIT_LENGTH * UNIT_TIME
                        static constexpr float_X SD = static_cast<float_X>(Params::SD_SI / UNIT_TIME / UNIT_LENGTH);
                        // unit: rad * UNIT_TIME
                        static constexpr float_X AD = static_cast<float_X>(Params::AD_SI / UNIT_TIME);
                        // unit: UNIT_TIME^2
                        static constexpr float_X GDD = static_cast<float_X>(Params::GDD_SI / UNIT_TIME / UNIT_TIME);
                        // unit: UNIT_TIME^3
                        static constexpr float_X TOD
                            = static_cast<float_X>(Params::TOD_SI / UNIT_TIME / UNIT_TIME / UNIT_TIME);

                        struct LaserParams
                        {
                            //! cell position in the internal laser coordinate system
                            float3_X internPos;
                            //! Distance from the current cell to the focus in laser propagation direction.
                            float_X cellToFocusDistance;
                            //! Center of a frequency's spatial distribution
                            float_X center;
                            //! frequency for which the E-value is calculated
                            float_X omega;
                            //! Initial frequency dependent complex phase
                            float_X alpha;

                            /**
                             * @tparam T_BaseFunctor laser type with base pulse knowledge
                             * @param totalCellIdx cell index in the total domain (including all moving window slides)
                             * @param Omega frequency for which the E-value is calculated
                             */
                            template<typename T_BaseFunctor>
                            HDINLINE LaserParams(
                                T_BaseFunctor const& baseFunctor,
                                floatD_X const& totalCellIdx,
                                float_X const Omega)
                                : omega(Omega)
                                , alpha(expandedWaveVectorX(Omega))
                            {
                                internPos = baseFunctor.getInternalCoordinates(totalCellIdx);

                                /* Calculate focus position relative to the current point in the propagation direction.
                                 * Coordinate system is PIConGPUs and not the laser internal coordinate system where X
                                 * is the propagation direction.
                                 */
                                float3_X const focusRelativeToOrigin
                                    = baseFunctor.getFocus() - baseFunctor.getOrigin();

                                /* Relative focus distance from the laser origin to the focus, transformed into the
                                 * laser coordination system where X is the propagation direction. */
                                float_X const distanceFocusRelativeToOrigin
                                    = pmacc::math::dot(focusRelativeToOrigin, baseFunctor.getAxis0());

                                cellToFocusDistance = distanceFocusRelativeToOrigin - internPos[0];

                                center = SD * (omega - Base::w)
                                    + SPEED_OF_LIGHT * alpha * cellToFocusDistance / (W0 * Base::w);
                            }

                            /** Helper function to calculate the electric field in frequency domain.
                             * Initial frequency dependent complex phase expanded up to third order in (Omega -
                             * Omega_0). Takes only first order angular dispersion d theta / d Omega = theta^prime into
                             * account and neglects all higher order angular dispersion terms, e.g. theta^{prime
                             * prime}, theta^{prime prime prime}, ...
                             *
                             * @param Omega frequency for which the E-value is calculated
                             */
                            HDINLINE float_X expandedWaveVectorX(float_X const Omega) const
                            {
                                return W0 / SPEED_OF_LIGHT
                                    * (Base::w * AD * (Omega - Base::w) + AD * (Omega - Base::w) * (Omega - Base::w)
                                       - Base::w / 6.0_X * AD * AD * AD * (Omega - Base::w) * (Omega - Base::w)
                                           * (Omega - Base::w));
                            }
                        };
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
                        /** The following two functions provide the electric field in frequency domain
                         * E(Omega) = amp * exp(-i*phi)
                         * Please ensure that E(Omega = 0) = 0 (no constant field contribution), i.e. the pulse
                         * length has to be big enough. Otherwise the implemented DFT will produce wrong results.
                         */
                        HDINLINE float_X amp(typename Unitless::LaserParams const laserParams) const
                        {
                            auto const focusPos = laserParams.cellToFocusDistance;
                            auto const Omega = laserParams.omega;
                            auto const pos = laserParams.internPos;
                            auto const center = laserParams.center;

                            // beam waist at the generation plane so that at focus we will get W0
                            float_X const waist = Unitless::W0
                                * math::sqrt(1.0_X
                                             + (focusPos / Unitless::rayleighLength)
                                                 * (focusPos / Unitless::rayleighLength));

                            // gaussian envelope in frequency domain
                            float_X mag = math::exp(
                                -(Omega - Unitless::w) * (Omega - Unitless::w) * Unitless::PULSE_DURATION
                                * Unitless::PULSE_DURATION);

                            // transversal envelope
                            mag *= math::exp(
                                -(pos[1] - center) * (pos[1] - center) / waist / waist); // envelope y - direction
                            mag *= math::exp(
                                -(pos[2] - center) * (pos[2] - center) / waist / waist); // envelope z - direction

                            // distinguish between dimensions
                            if constexpr(simDim == DIM2)
                            {
                                // pos has just two entries: pos[0] as propagation direction and pos[1] as
                                // transversal direction
                                mag *= math::sqrt(Unitless::W0 / waist);
                            }
                            else if constexpr(simDim == DIM3)
                            {
                                mag *= Unitless::W0 / waist;
                            }

                            // Normalization to Amplitude
                            mag *= math::sqrt(pmacc::math::Pi<float_X>::doubleValue * 0.5_X) * 2.0_X
                                * Unitless::PULSE_DURATION * Unitless::AMPLITUDE;

                            // Dividing amplitude by 2 to compensate doubled spectral field strength
                            // resulting from E(-Omega) = E*(Omega), which has to be fulfilled for E(t) to be real
                            mag *= 0.5_X;

                            return mag;
                        }

                        HDINLINE float_X
                        phi(typename Unitless::LaserParams const laserParams, float_X const phaseShift) const
                        {
                            auto const focusPos = laserParams.cellToFocusDistance;
                            auto const Omega = laserParams.omega;
                            auto const pos = laserParams.internPos;
                            auto const center = laserParams.center;
                            auto const alpha = laserParams.alpha;

                            // inverse radius of curvature of the pulse's wavefronts
                            auto const R_inv = -focusPos
                                / (Unitless::rayleighLength * Unitless::rayleighLength + focusPos * focusPos);
                            // the Gouy phase shift
                            auto const xi = math::atan(-focusPos / Unitless::rayleighLength);

                            // shifting pulse for half of INIT_TIME to start with the front of the laser pulse
                            constexpr auto mue = 0.5_X * Unitless::INIT_TIME;
                            float_X const timeDelay = mue + focusPos / SPEED_OF_LIGHT;

                            float_X phase = -Omega * focusPos / SPEED_OF_LIGHT
                                + 0.5_X * Unitless::GDD * (Omega - Unitless::w) * (Omega - Unitless::w)
                                + Unitless::TOD / 6.0_X * (Omega - Unitless::w) * (Omega - Unitless::w)
                                    * (Omega - Unitless::w)
                                + phaseShift + Unitless::LASER_PHASE + Omega * timeDelay;

                            phase += ((pos[1] - center) * (pos[1] - center) + (pos[2] - center) * (pos[2] - center))
                                * Omega * 0.5_X * R_inv / SPEED_OF_LIGHT;
                            phase -= alpha * (pos[1] + pos[2]) / Unitless::W0;

                            // distinguish between dimensions
                            if constexpr(simDim == DIM2)
                            {
                                phase += alpha * alpha / 4.0_X * focusPos / Unitless::rayleighLength;
                                phase -= 0.5_X * xi;
                            }
                            else if constexpr(simDim == DIM3)
                            {
                                phase += alpha * alpha / 2.0_X * focusPos / Unitless::rayleighLength;
                                phase -= xi;
                            }
                            return phase;
                        }

                        /** Get value of E field in time domain for the given position, using DFT
                         * Interpolation order of DFT given via timestep in grid.param and INIT_TIME
                         * Neglecting the constant part of DFT (k = 0) because there should be no constant field
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @param phaseShift additional phase shift to add on top of everything else,
                         *                   in radian
                         */
                        HDINLINE float_X getValueE(floatD_X const& totalCellIdx, float_X const phaseShift) const
                        {
                            auto const time = this->getCurrentTime(totalCellIdx);
                            if(time < 0.0_X)
                                return 0.0_X;

                            // turning Laser off after producing one pulse (during INIT_TIME) to avoid periodic pulses
                            else if(time > Unitless::INIT_TIME)
                                return 0.0_X;

                            // interpolation order
                            float_X N_raw = Unitless::INIT_TIME / DELTA_T;
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
                            int const center_k
                                = static_cast<int>(SPEED_OF_LIGHT * Unitless::INIT_TIME / Unitless::Base::WAVE_LENGTH);

                            // index of the lowest frequency in the Gaussian distributed spectrum which is used in the
                            // DFT 4*sigma_Om distance from central frequency
                            int const minOm_k
                                = center_k - static_cast<int>(Unitless::SPECTRAL_SUPPORT * sigma_Om / dOmk);
                            int const k_min = math::max(minOm_k, 1);

                            // index of the highest frequency in the Gaussian distributed spectrum which is used in the
                            // DFT
                            int const maxOm_k = 2 * center_k - minOm_k;
                            int const k_max = math::min(maxOm_k, n);

                            // electric field in time-domain
                            float_X E_t = 0.0_X;

                            for(int k = k_min; k <= k_max; k++)
                            {
                                // stores angular frequency for DFT-loop
                                float_X const Omk = static_cast<float_X>(k) * dOmk;
                                using LaserParams = typename Unitless::LaserParams;
                                auto laserParams = LaserParams(static_cast<Base>(*this), totalCellIdx, Omk);

                                float_X sinPhi, cosPhi;
                                float_X sinOmkt, cosOmkt;
                                pmacc::math::sincos(phi(laserParams, phaseShift), sinPhi, cosPhi);
                                pmacc::math::sincos(laserParams.omega * time, sinOmkt, cosOmkt);
                                E_t += 2.0_X * amp(laserParams) * (cosPhi * cosOmkt + sinPhi * sinOmkt) / DELTA_T;
                            }

                            E_t /= static_cast<float_X>(2 * n + 1); // Normalization from DFT

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

            namespace detail
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
                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::DispersivePulse<T_Params>>::type>;
                };

            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
