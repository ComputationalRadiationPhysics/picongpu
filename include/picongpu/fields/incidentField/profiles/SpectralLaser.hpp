/* Copyright 2022 Fabia Dietrich
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/incidentField/Functors.hpp"
#include "picongpu/fields/incidentField/Traits.hpp"

#include <pmacc/algorithms/math/defines/pi.hpp>

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

                    /** Unitless spectralLaser parameters
                     *
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct SpectralLaserUnitless
                        : public BaseParamUnitless<T_Params>
                    {
                        //! User SI parameters
                        using Params = T_Params;

                        //! Base unitless parameters
                        using Base = BaseParamUnitless<T_Params>;

                        // unit: UNIT_LENGTH
                        static constexpr float_X W0 = static_cast<float_X>(Params::W0_SI / UNIT_LENGTH);

                        // rayleigh length in propagation direction
                        static constexpr float_X R = pmacc::math::Pi<float_X>::value * W0 * W0 / Base::WAVE_LENGTH;

                        // unit: UNIT_TIME
                        // corresponds to period length of DFT
                        static constexpr float_X INIT_TIME = static_cast<float_X>(Params::PULSE_INIT) * Base::PULSE_LENGTH;

                        // Dispersion parameters
                        // unit: UNIT_TIME^2
                        static constexpr float_X GDD = static_cast<float_X>(Params::GDD_SI / UNIT_TIME / UNIT_TIME);
                        // unit: UNIT_TIME^3
                        static constexpr float_X TOD = static_cast<float_X>(Params::TOD_SI / UNIT_TIME / UNIT_TIME / UNIT_TIME);

                    };

                    /** SpectralLaser incident E functor
                     *
                     * @tparam T_Params parameters
                     */
                    template<typename T_Params>
                    struct SpectralLaserFunctorIncidentE
                        : public SpectralLaserUnitless<T_Params>
                        , public incidentField::detail::BaseFunctorE<T_Params>
                    {
                        //! Unitless parameters type
                        using Unitless = SpectralLaserUnitless<T_Params>;

                        //! Base functor type
                        using Base = incidentField::detail::BaseFunctorE<T_Params>;

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE SpectralLaserFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
                            : Base(currentStep, unitField)
                        {

                        }

                        /** Calculate incident field E value for the given position
                         *
                         * Given th e E-Field in dependence of Frequency, we want to calculate its value for a specific time
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @return incident field E value in internal units
                         */
                        HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
                        {
                            if(Unitless::Polarisation == PolarisationType::Linear)
                                return this->getLinearPolarizationVector() * getValue(totalCellIdx, 0.0_X);
                            else
                            {
                                auto const phaseShift = pmacc::math::Pi<float_X>::halfValue;
                                return this->getCircularPolarizationVector1() * getValue(totalCellIdx, phaseShift)
                                    + this->getCircularPolarizationVector2() * getValue(totalCellIdx, 0.0_X);
                            }
                        }

                        /** The following two functions provide the electric field in frequency domain
                        * E(Omega) = E_amp * exp(-i*phi)
                        *
                        * @param totalCellIdx cell index in the total domain (including all moving window slides)
                        * @param Omega frequency for which the E-value is wanted
                        */
                        HDINLINE float_X E_amp(floatD_X const& totalCellIdx, float_X const Omega) const
                        {
                            // transform to 3d internal coordinate system
                            float3_X pos = this->getInternalCoordinates(totalCellIdx);

                            // calculate focus position relative to the current point in the propagation direction
                            auto const focusRelativeToOrigin = float3_X(
                                                                   Unitless::FOCUS_POSITION_X,
                                                                   Unitless::FOCUS_POSITION_Y,
                                                                   Unitless::FOCUS_POSITION_Z)
																   - this->origin;
                            // current distance to focus position
                            float_X const focusPos = math::sqrt(pmacc::math::abs2(focusRelativeToOrigin)) - pos[0];
                            // beam waist at the generation plane so that at focus we will get W0
                            float_X const waist = Unitless::W0
                                * math::sqrt(1.0_X + (focusPos / Unitless::R) * (focusPos / Unitless::R));

                            auto planeNoNormal = float3_X::create(1.0_X);
                            planeNoNormal[0] = 0.0_X;
                            auto const transversalDistanceSquared = pmacc::math::abs2(pos * planeNoNormal);

                            auto const r2OverW2 = transversalDistanceSquared / waist / waist;

                            // central frequency Unitless::w
                            // gaussian envelope in frequency domain
                            // using pulse length - bandwidth product: sigma_Omega = sqrt(2)/tau
                            float_X norm = math::sqrt(pmacc::math::Pi<float_X>::halfValue * 2.0_X) * Unitless::PULSE_LENGTH * 2.0_X;
                            float_X eps = norm * math::exp(-(Omega - Unitless::w) * (Omega - Unitless::w) * Unitless::PULSE_LENGTH * Unitless::PULSE_LENGTH);
                            
                            float_X amp = eps * math::exp(-r2OverW2);

                            // distinguish between dimensions
                            if(simDim == DIM2)
                                amp *= math::sqrt(Unitless::W0 / waist);
                            else if(simDim == DIM3)
                                amp *= Unitless::W0 / waist;
                            return amp;
                        }
                         
                        HDINLINE float_X phi(floatD_X const& totalCellIdx, float_X const Omega) const
                        {
							// transform to 3d internal coordinate system
                            float3_X pos = this->getInternalCoordinates(totalCellIdx);

                            // calculate focus position relative to the current point in the propagation direction
                            auto const focusRelativeToOrigin = float3_X(
                                                                   Unitless::FOCUS_POSITION_X,
                                                                   Unitless::FOCUS_POSITION_Y,
                                                                   Unitless::FOCUS_POSITION_Z)
                                - this->origin;
                            // current distance to focus position
                            float_X const focusPos = math::sqrt(pmacc::math::abs2(focusRelativeToOrigin)) - pos[0];

                            auto planeNoNormal = float3_X::create(1.0_X);
                            planeNoNormal[0] = 0.0_X;
                            auto const transversalDistanceSquared = pmacc::math::abs2(pos * planeNoNormal);

                            // inverse radius of curvature of the beam's  wavefronts
                            auto const R_inv = -focusPos / (Unitless::R * Unitless::R + focusPos * focusPos);
                            // the Gouy phase shift
                            auto const xi = math::atan(-focusPos / Unitless::R);
                            auto const r = 0.5_X * transversalDistanceSquared * R_inv;

                            // central frequency Unitless::w
                            float_X phase = Omega / SPEED_OF_LIGHT * (r - focusPos) 
                                            + 0.5_X * Unitless::GDD * (Omega - Unitless::w) * (Omega - Unitless::w)
                                            + Unitless::TOD * (Omega - Unitless::w) * (Omega - Unitless::w) * (Omega - Unitless::w)/6.0_X;

                            // distinguish between dimensions
                            if(simDim == DIM2)
                                phase -= 0.5_X * xi;
                            else if(simDim == DIM3)
                                phase -= xi;
                            return phase;
						}
					private:
                        /** Get value for the given position, using DFT
                         * Interpolation order of DFT given via timestep in grid.param and INIT_TIME
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @param phaseShift additional phase shift to add on top of everything else,
                         *                   in radian
                         */
                        HDINLINE float_X getValue(floatD_X const& totalCellIdx, float_X const phaseShift) const
                        {
                            auto const time = this->getCurrentTime(totalCellIdx);
                            if(time < 0.0_X)
                                return 0.0_X;

                            // turning Laser off after producing a pulse
                            // e.g. initializing the pulse just during INIT_TIME
                            // otherwise DFT will lead to periodic pulses
                            else if(time > Unitless::INIT_TIME)
								return 0.0_X;
								
							// transform to 3d internal coordinate system
                            float3_X pos = this->getInternalCoordinates(totalCellIdx);
                            // calculate focus position relative to the current point in the propagation direction
                            auto const focusRelativeToOrigin = float3_X(
                                                     Unitless::FOCUS_POSITION_X,
                                                     Unitless::FOCUS_POSITION_Y,
                                                     Unitless::FOCUS_POSITION_Z)
                                                     - this->origin;
                            // current distance to focus position
                            float_X const focusPos = math::sqrt(pmacc::math::abs2(focusRelativeToOrigin)) - pos[0];
                            
                            // timestep also in UNIT_TIME
                            float_X const dt = static_cast<float_X>(picongpu::SI::DELTA_T_SI / UNIT_TIME);
                            // interpolation order
                            float_X N_raw = Unitless::INIT_TIME / dt;
                            int n = static_cast<int>((N_raw) / 2);  // -0 instead of -1 for rounding up N_raw
                            
                            float_X E_t = 0.0_X;    // electric field in time-domain
                            float_X Omk = 0.0_X;    // angular frequency for DFT-loop; Omk= 2pi*k/T
                            float_X phase = 0.0_X;  // Laser phase in time domain
                            // shifting pulse for half of INIT_TIME to start with the front of the laser pulse
                            constexpr auto mue = 0.5_X * Unitless::INIT_TIME;

                            for(int k=1; k<n+1; k++)
                            {
                                Omk = static_cast<float_X>(k) * pmacc::math::Pi<float_X>::doubleValue / Unitless::INIT_TIME;
                                phase = Omk * (time - mue - focusPos / SPEED_OF_LIGHT) + Unitless::LASER_PHASE + phaseShift;
                                E_t += (E_amp(totalCellIdx, Omk) * math::cos(phi(totalCellIdx, Omk)) / dt * math::cos(phase)
                                      + E_amp(totalCellIdx, Omk) * math::sin(phi(totalCellIdx, Omk)) / dt * math::sin(phase));
                            }

                            E_t /= static_cast<float_X>(2*n + 1);  // Normalization from DFT

                            // Normalization to Amplitude
                            E_t *= Unitless::AMPLITUDE;

                            return E_t;
                        }
                    }; // SpectralLaserFunctorIncidentE
                } // namespace detail
                
                template<typename T_Params>
                struct SpectralLaser
                {
                    //! Get text name of the incident field profile
                    static HINLINE std::string getName()
                    {
                        return "SpectralLaser";
                    }
                };
                
            } // namespace profiles

            namespace detail
            {
                /** Get type of incident field E functor for the spectral laser profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentE<profiles::SpectralLaser<T_Params>>
                {
                    using type = profiles::detail::SpectralLaserFunctorIncidentE<T_Params>;
                };

                /** Get type of incident field B functor for the spectral laser profile type
                 *
                 * Rely on SVEA to calculate value of B from E.
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentB<profiles::SpectralLaser<T_Params>>
                {
                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::SpectralLaser<T_Params>>::type>;
                };
                
            } // namespace detail            
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu

