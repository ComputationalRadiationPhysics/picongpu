/* Copyright 2018-2022 Ilja Goethel, Axel Huebl, Sergei Bastrakov
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
                    /** Unitless exponential ramp with prepulse parameters
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct ExpRampWithPrepulseUnitless : public T_Params
                    {
                        using Params = T_Params;

                        static constexpr float_X WAVE_LENGTH
                            = static_cast<float_X>(Params::WAVE_LENGTH_SI / UNIT_LENGTH); // unit: meter
                        static constexpr float_X PULSE_LENGTH
                            = static_cast<float_X>(Params::PULSE_LENGTH_SI / UNIT_TIME); // unit: seconds (1 sigma)
                        static constexpr float_X LASER_NOFOCUS_CONSTANT
                            = static_cast<float_X>(Params::LASER_NOFOCUS_CONSTANT_SI / UNIT_TIME); // unit: seconds
                        static constexpr float_X AMPLITUDE
                            = static_cast<float_X>(Params::AMPLITUDE_SI / UNIT_EFIELD); // unit: Volt /meter
                        static constexpr float_X W0_AXIS_1
                            = static_cast<float_X>(Params::W0_AXIS_1_SI / UNIT_LENGTH); // unit: meter
                        static constexpr float_X W0_AXIS_2
                            = static_cast<float_X>(Params::W0_AXIS_2_SI / UNIT_LENGTH); // unit: meter

                        static constexpr float_64 TIME_PREPULSE
                            = static_cast<float_64>(Params::TIME_PREPULSE_SI / UNIT_TIME);
                        static constexpr float_64 TIME_PEAKPULSE
                            = static_cast<float_64>(Params::TIME_PEAKPULSE_SI / UNIT_TIME);
                        static constexpr float_64 TIME_1 = static_cast<float_64>(Params::TIME_POINT_1_SI / UNIT_TIME);
                        static constexpr float_64 TIME_2 = static_cast<float_64>(Params::TIME_POINT_2_SI / UNIT_TIME);
                        static constexpr float_64 TIME_3 = static_cast<float_64>(Params::TIME_POINT_3_SI / UNIT_TIME);
                        static constexpr float_X endUpramp = TIME_PEAKPULSE - 0.5_X * LASER_NOFOCUS_CONSTANT;
                        static constexpr float_X startDownramp = TIME_PEAKPULSE + 0.5_X * LASER_NOFOCUS_CONSTANT;

                        static constexpr float_X INIT_TIME
                            = static_cast<float_X>((TIME_PEAKPULSE + Params::RAMP_INIT * PULSE_LENGTH) / UNIT_TIME);

                        // compile-time checks for physical sanity:
                        static_assert(
                            (TIME_1 < TIME_2) && (TIME_2 < TIME_3) && (TIME_3 < endUpramp),
                            "The times in the parameters TIME_POINT_1/2/3 and the beginning of the plateau (which is "
                            "at "
                            "TIME_PEAKPULSE - 0.5*RAMP_INIT*PULSE_LENGTH) should be in ascending order");

                        // some prerequisites for check of intensities (approximate check, because I can't use exp and
                        // log)
                        static constexpr float_X ratio_dt
                            = (endUpramp - TIME_3) / (TIME_3 - TIME_2); // ratio of time intervals
                        static constexpr float_X ri1
                            = Params::INT_RATIO_POINT_3 / Params::INT_RATIO_POINT_2; // first intensity ratio
                        static constexpr float_X ri2
                            = 0.2_X / Params::INT_RATIO_POINT_3; // second intensity ratio (0.2 is an arbitrary upper
                                                                 // border for the intensity of the exp ramp)

                        /* Approximate check, if ri1 ^ ratio_dt > ri2. That would mean, that the exponential curve
                         * through (time2, int2) and (time3, int3) lies above (endUpramp, 0.2) the power function is
                         * emulated by "rounding" the exponent to a rational number and expanding both sides by the
                         * common denominator, to get integer powers, see below for this, the range for ratio_dt is
                         * split into parts; the checked condition is "rounded down", i.e. it's weaker in every point
                         * of those ranges except one.
                         */
                        static constexpr bool intensity_too_big = (ratio_dt >= 3._X && ri1 * ri1 * ri1 > ri2)
                            || (ratio_dt >= 2._X && ri1 * ri1 > ri2)
                            || (ratio_dt >= 1.5_X && ri1 * ri1 * ri1 > ri2 * ri2) || (ratio_dt >= 1._X && ri1 > ri2)
                            || (ratio_dt >= 0.8_X && ri1 * ri1 * ri1 * ri1 > ri2 * ri2 * ri2 * ri2 * ri2)
                            || (ratio_dt >= 0.75_X && ri1 * ri1 * ri1 > ri2 * ri2 * ri2 * ri2)
                            || (ratio_dt >= 0.67_X && ri1 * ri1 > ri2 * ri2 * ri2)
                            || (ratio_dt >= 0.6_X && ri1 * ri1 * ri1 > ri2 * ri2 * ri2 * ri2 * ri2)
                            || (ratio_dt >= 0.5_X && ri1 > ri2 * ri2)
                            || (ratio_dt >= 0.4_X && ri1 * ri1 > ri2 * ri2 * ri2 * ri2 * ri2)
                            || (ratio_dt >= 0.33_X && ri1 > ri2 * ri2 * ri2)
                            || (ratio_dt >= 0.25_X && ri1 > ri2 * ri2 * ri2 * ri2)
                            || (ratio_dt >= 0.2_X && ri1 > ri2 * ri2 * ri2 * ri2 * ri2);
                        static_assert(
                            !intensity_too_big,
                            "The intensities of the ramp are very large - the extrapolation to the time of the main "
                            "pulse "
                            "would give more than half of the pulse amplitude. This is not a Gaussian pulse at all "
                            "anymore - probably some of the parameters are different from what you think!?");

                        /* a symmetric pulse will be initialized at position z=0 for
                         * a time of RAMP_INIT * PULSE_LENGTH + LASER_NOFOCUS_CONSTANT = INIT_TIME.
                         * we shift the complete pulse for the half of this time to start with
                         * the front of the laser pulse.
                         */
                        static constexpr float_X time_start_init
                            = static_cast<float_X>(TIME_1 - (0.5 * Params::RAMP_INIT * PULSE_LENGTH));
                        static constexpr float_64 f = SPEED_OF_LIGHT / WAVE_LENGTH;
                        static constexpr float_64 w = 2.0 * PI * f;
                    };

                    /** Exponential ramp with prepulse incident E functor
                     *
                     * @tparam T_Params parameters
                     * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                     * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from
                     * the max boundary inwards)
                     */
                    template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                    struct ExpRampWithPrepulseFunctorIncidentE
                        : public ExpRampWithPrepulseUnitless<T_Params>
                        , public BaseFunctorE<T_axis, T_direction>
                    {
                        //! Unitless parameters type
                        using Unitless = ExpRampWithPrepulseUnitless<T_Params>;

                        //! Base functor type
                        using Base = BaseFunctorE<T_axis, T_direction>;

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE ExpRampWithPrepulseFunctorIncidentE(
                            float_X const currentStep,
                            float3_64 const unitField)
                            : Base(unitField)
                            , elong(getLongitudinal(currentStep))
                        {
                            auto const& subGrid = Environment<simDim>::get().SubGrid();
                            totalDomainCells = precisionCast<float_X>(subGrid.getTotalDomain().size);
                        }

                        /** Calculate incident field E value for the given position
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @return incident field E value in internal units
                         */
                        HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
                        {
                            return elong * getTransversal(totalCellIdx);
                        }

                    private:
                        //! Total domain size in cells
                        floatD_X totalDomainCells;

                        //! Precalulated time-dependent longitudinal value
                        float3_X const elong;

                        //! Get time-dependent longitudinal vector factor
                        HINLINE float3_X getLongitudinal(float_X const currentStep) const
                        {
                            /* initialize the laser not in the first cell is equal to a negative shift
                             * in time
                             */
                            float_64 const runTime = Unitless::time_start_init + DELTA_T * currentStep;
                            float_64 const phase = Unitless::w * runTime + Unitless::LASER_PHASE;
                            auto result = float3_64::create(0.0);
                            if(Unitless::Polarisation == Unitless::LINEAR_AXIS_2)
                            {
                                result[Base::dir2] = math::sin(phase);
                            }
                            else if(Unitless::Polarisation == Unitless::LINEAR_AXIS_1)
                            {
                                result[Base::dir1] = math::sin(phase);
                            }
                            else if(Unitless::Polarisation == Unitless::CIRCULAR)
                            {
                                result[Base::dir2] = math::sin(phase) / math::sqrt(2.0_X);
                                result[Base::dir1] = math::cos(phase) / math::sqrt(2.0_X);
                            }
                            return precisionCast<float_X>(result * getEnvelope(runTime));
                        }

                        HINLINE float_64 getEnvelope(float_64 const runTime) const
                        {
                            /* workaround for clang 5 linker issues
                             * `undefined reference to
                             * `picongpu::fields::laserProfiles::ExpRampWithPrepulseParam::INT_RATIO_POINT_1'`
                             */
                            constexpr auto int_ratio_prepule = Unitless::INT_RATIO_PREPULSE;
                            constexpr auto int_ratio_point_1 = Unitless::INT_RATIO_POINT_1;
                            constexpr auto int_ratio_point_2 = Unitless::INT_RATIO_POINT_2;
                            constexpr auto int_ratio_point_3 = Unitless::INT_RATIO_POINT_3;
                            auto const AMP_PREPULSE = math::sqrt(int_ratio_prepule) * Unitless::AMPLITUDE;
                            auto const AMP_1 = math::sqrt(int_ratio_point_1) * Unitless::AMPLITUDE;
                            auto const AMP_2 = math::sqrt(int_ratio_point_2) * Unitless::AMPLITUDE;
                            auto const AMP_3 = math::sqrt(int_ratio_point_3) * Unitless::AMPLITUDE;

                            auto env = 0.0;
                            bool const before_preupramp = runTime < Unitless::time_start_init;
                            bool const before_start = runTime < Unitless::TIME_1;
                            bool const before_peakpulse = runTime < Unitless::endUpramp;
                            bool const during_first_exp = (Unitless::TIME_1 < runTime) && (runTime < Unitless::TIME_2);
                            bool const after_peakpulse = Unitless::startDownramp <= runTime;

                            if(before_preupramp)
                                env = 0.;
                            else if(before_start)
                            {
                                env = AMP_1 * gauss(runTime - Unitless::TIME_1);
                            }
                            else if(before_peakpulse)
                            {
                                float_X const ramp_when_peakpulse = extrapolateExpo(
                                                                        Unitless::TIME_2,
                                                                        AMP_2,
                                                                        Unitless::TIME_3,
                                                                        AMP_3,
                                                                        Unitless::endUpramp)
                                    / Unitless::AMPLITUDE;

                                if(ramp_when_peakpulse > 0.5)
                                {
                                    log<picLog::PHYSICS>(
                                        "Attention, the intensities of the laser upramp are very large! "
                                        "The extrapolation of the last exponential to the time of "
                                        "the peakpulse gives more than half of the amplitude of "
                                        "the peak Gaussian. This is not a Gaussian at all anymore, "
                                        "and physically very unplausible, check the params for misunderstandings!");
                                }

                                env += Unitless::AMPLITUDE * (1._X - ramp_when_peakpulse)
                                    * gauss(runTime - Unitless::endUpramp);
                                env += AMP_PREPULSE * gauss(runTime - Unitless::TIME_PREPULSE);
                                if(during_first_exp)
                                    env += extrapolateExpo(Unitless::TIME_1, AMP_1, Unitless::TIME_2, AMP_2, runTime);
                                else
                                    env += extrapolateExpo(Unitless::TIME_2, AMP_2, Unitless::TIME_3, AMP_3, runTime);
                            }
                            else if(!after_peakpulse)
                                env = Unitless::AMPLITUDE;
                            else // after startDownramp
                                env = Unitless::AMPLITUDE * gauss(runTime - Unitless::startDownramp);
                            return env;
                        }

                        /** takes time t relative to the center of the Gaussian and returns value
                         * between 0 and 1, i.e. as multiple of the max value.
                         * use as: amp_t = amp_0 * gauss( t - t_0 )
                         */
                        HINLINE float_64 gauss(float_64 const t) const
                        {
                            auto const exponent = t / Unitless::PULSE_LENGTH;
                            return math::exp(-0.25 * exponent * exponent);
                        }

                        /** get value of exponential curve through two points at given t
                         * t/t1/t2 given as float_X, since the envelope doesn't need the accuracy
                         */
                        HINLINE float_64 extrapolateExpo(
                            float_64 const t1,
                            float_64 const a1,
                            float_64 const t2,
                            float_64 const a2,
                            float_64 const t) const
                        {
                            auto const log1 = (t2 - t) * math::log(a1);
                            auto const log2 = (t - t1) * math::log(a2);
                            return math::exp((log1 + log2) / (t2 - t1));
                        }

                        //! Get position-dependent transversal scalar factor
                        HDINLINE float_X getTransversal(const floatD_X& totalCellIdx) const
                        {
                            floatD_X transversalPosition
                                = (totalCellIdx - totalDomainCells * 0.5_X) * cellSize.shrink<simDim>();
                            transversalPosition[Base::dir0] = 0.0_X;
                            auto w0 = float3_X::create(1.0_X);
                            w0[Base::dir1] = Unitless::W0_AXIS_1;
                            w0[Base::dir2] = Unitless::W0_AXIS_2;
                            float_X const r2 = pmacc::math::abs2(transversalPosition / w0.shrink<simDim>());
                            return math::exp(-r2);
                        }
                    };
                } // namespace detail
            } // namespace profiles

            namespace detail
            {
                /** Get type of incident field E functor for the exponential ramp with prepulse profile type
                 *
                 * @tparam T_Params parameters
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 */
                template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                struct GetFunctorIncidentE<profiles::ExpRampWithPrepulse<T_Params>, T_axis, T_direction>
                {
                    using type = profiles::detail::ExpRampWithPrepulseFunctorIncidentE<T_Params, T_axis, T_direction>;
                };

                /** Get type of incident field B functor for the exponential ramp with prepulse  profile type
                 *
                 * Rely on SVEA to calculate value of B from E.
                 *
                 * @tparam T_Params parameters
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 */
                template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                struct GetFunctorIncidentB<profiles::ExpRampWithPrepulse<T_Params>, T_axis, T_direction>
                {
                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::ExpRampWithPrepulse<T_Params>, T_axis, T_direction>::
                            type>;
                };
            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
