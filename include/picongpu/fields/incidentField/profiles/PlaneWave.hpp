/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch, Sergei Bastrakov
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
#include "picongpu/fields/incidentField/profiles/BaseFunctorE.hpp"

#include <cstdint>

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
                    /** Unitless plane wave parameters
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct PlaneWaveUnitless : public T_Params
                    {
                        using Params = T_Params;

                        static constexpr float_X WAVE_LENGTH
                            = float_X(Params::WAVE_LENGTH_SI / UNIT_LENGTH); // unit: meter
                        static constexpr float_X PULSE_LENGTH
                            = float_X(Params::PULSE_LENGTH_SI / UNIT_TIME); // unit: seconds (1 sigma)
                        static constexpr float_X LASER_NOFOCUS_CONSTANT
                            = float_X(Params::LASER_NOFOCUS_CONSTANT_SI / UNIT_TIME); // unit: seconds
                        static constexpr float_X AMPLITUDE
                            = float_X(Params::AMPLITUDE_SI / UNIT_EFIELD); // unit: Volt /meter
                        static constexpr float_X INIT_TIME = float_X(
                            (Params::RAMP_INIT * Params::PULSE_LENGTH_SI + Params::LASER_NOFOCUS_CONSTANT_SI)
                            / UNIT_TIME); // unit: seconds (full inizialisation length)
                        static constexpr float_64 f = SPEED_OF_LIGHT / WAVE_LENGTH;
                    };

                    /** Plane wave incident E functor
                     *
                     * @tparam T_Params parameters
                     * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                     * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from
                     * the max boundary inwards)
                     */
                    template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                    struct PlaneWaveFunctorIncidentE
                        : public PlaneWaveUnitless<T_Params>
                        , public BaseFunctorE<T_axis, T_direction>
                    {
                        //! Unitless parameters type
                        using Unitless = PlaneWaveUnitless<T_Params>;

                        //! Base functor type
                        using Base = BaseFunctorE<T_axis, T_direction>;

                        /** Create a functor on the host side
                         *
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE PlaneWaveFunctorIncidentE(const float3_64 unitField) : Base(unitField)
                        {
                        }

                        /** Calculate incident field E value for the given position and time.
                         *
                         * Since it is a plane wave parallel to the generating surface, the value only depends on time.
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @param currentStep current time step index, note that it is fractional
                         * @return incident field E value in internal units
                         */
                        HDINLINE float3_X operator()(const floatD_X& /*totalCellIdx*/, const float_X currentStep) const
                        {
                            float_64 const runTime = DELTA_T * currentStep;
                            float_64 envelope = float_64(Unitless::AMPLITUDE);
                            float_64 const mue = 0.5 * Unitless::RAMP_INIT * Unitless::PULSE_LENGTH;
                            float_64 const w = 2.0 * PI * Unitless::f;
                            float_64 const tau = Unitless::PULSE_LENGTH * math::sqrt(2.0);
                            float_64 const endUpramp = mue;
                            float_64 const startDownramp = mue + Unitless::LASER_NOFOCUS_CONSTANT;
                            float_64 integrationCorrectionFactor = 0.0;
                            if(runTime > startDownramp)
                            {
                                // downramp = end
                                float_64 const exponent = (runTime - startDownramp) / tau;
                                envelope *= exp(-0.5 * exponent * exponent);
                                integrationCorrectionFactor = (runTime - startDownramp) / (w * tau * tau);
                            }
                            else if(runTime < endUpramp)
                            {
                                // upramp = start
                                float_64 const exponent = (runTime - endUpramp) / tau;
                                envelope *= exp(-0.5 * exponent * exponent);
                                integrationCorrectionFactor = (runTime - endUpramp) / (w * tau * tau);
                            }

                            float_64 const timeOszi = runTime - endUpramp;
                            float_64 const t_and_phase = w * timeOszi + Unitless::LASER_PHASE;
                            // to understand both components [sin(...) + t/tau^2 * cos(...)] see description above
                            auto const baseValue = static_cast<float_X>(
                                envelope
                                * (math::sin(t_and_phase) + math::cos(t_and_phase) * integrationCorrectionFactor));
                            auto elong = float3_X::create(0.0_X);
                            if(Unitless::Polarisation == Unitless::LINEAR_AXIS_1)
                            {
                                elong[Base::dir1] = baseValue;
                            }
                            else if(Unitless::Polarisation == Unitless::LINEAR_AXIS_2)
                            {
                                elong[Base::dir2] = baseValue;
                            }
                            else if(Unitless::Polarisation == Unitless::CIRCULAR)
                            {
                                elong[Base::dir1] = baseValue / math::sqrt(2.0_X);
                                elong[Base::dir2] = baseValue / math::sqrt(2.0_X);
                            }
                            return elong;
                        }
                    };
                } // namespace detail
            } // namespace profiles

            namespace detail
            {
                /** Get type of incident field E functor for the plane wave profile type
                 *
                 * @tparam T_Params parameters
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 */
                template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                struct GetFunctorIncidentE<profiles::PlaneWave<T_Params>, T_axis, T_direction>
                {
                    using type = profiles::detail::PlaneWaveFunctorIncidentE<T_Params, T_axis, T_direction>;
                };

                /** Get type of incident field E functor for the plane wave profile type
                 *
                 * For plane wave there is no difference between directly- and SVEA-calculating B, so reuse SVEA for
                 * brevity.
                 *
                 * @tparam T_Params parameters
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 */
                template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                struct GetFunctorIncidentB<profiles::PlaneWave<T_Params>, T_axis, T_direction>
                {
                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::PlaneWave<T_Params>, T_axis, T_direction>::type,
                        T_axis,
                        T_direction>;
                };
            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
