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

#include <pmacc/algorithms/math/defines/pi.hpp>

#include <cstdint>
#include <string>


namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace profiles
            {
                template<typename T_Params>
                struct ExpRampWithPrepulse
                {
                    //! Get text name of the incident field profile
                    static HINLINE std::string getName()
                    {
                        return "ExpRampWithPrepulse";
                    }
                };

                namespace detail
                {
                    /** Unitless exponential ramp with prepulse parameters
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct ExpRampWithPrepulseUnitless : public BaseTransversalGaussianParamUnitless<T_Params>
                    {
                        //! User SI parameters
                        using Params = T_Params;

                        //! Base unitless parameters
                        using Base = BaseTransversalGaussianParamUnitless<T_Params>;

                        // unit: UNIT_TIME
                        static constexpr float_X LASER_NOFOCUS_CONSTANT
                            = static_cast<float_X>(Params::LASER_NOFOCUS_CONSTANT_SI / UNIT_TIME);

                        static constexpr float_X TIME_PREPULSE
                            = static_cast<float_X>(Params::TIME_PREPULSE_SI / UNIT_TIME);
                        static constexpr float_X TIME_PEAKPULSE
                            = static_cast<float_X>(Params::TIME_PEAKPULSE_SI / UNIT_TIME);
                        static constexpr float_X TIME_1 = static_cast<float_X>(Params::TIME_POINT_1_SI / UNIT_TIME);
                        static constexpr float_X TIME_2 = static_cast<float_X>(Params::TIME_POINT_2_SI / UNIT_TIME);
                        static constexpr float_X TIME_3 = static_cast<float_X>(Params::TIME_POINT_3_SI / UNIT_TIME);
                        static constexpr float_X endUpramp = TIME_PEAKPULSE - 0.5_X * LASER_NOFOCUS_CONSTANT;
                        static constexpr float_X startDownramp = TIME_PEAKPULSE + 0.5_X * LASER_NOFOCUS_CONSTANT;

                        static constexpr float_X INIT_TIME = static_cast<float_X>(
                            (TIME_PEAKPULSE + Params::RAMP_INIT * Base::PULSE_LENGTH) / UNIT_TIME);

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

                        /* a symmetric pulse will be initialized at generation plane for
                         * a time of RAMP_INIT * PULSE_LENGTH + LASER_NOFOCUS_CONSTANT = INIT_TIME.
                         * we shift the complete pulse for the half of this time to start with
                         * the front of the laser pulse.
                         */
                        static constexpr float_X time_start_init
                            = static_cast<float_X>(TIME_1 - (0.5_X * Params::RAMP_INIT * Base::PULSE_LENGTH));
                    };

                    /** Exponential ramp with prepulse incident E functor
                     *
                     * @tparam T_Params parameters
                     */
                    template<typename T_Params>
                    struct ExpRampWithPrepulseFunctorIncidentE
                        : public ExpRampWithPrepulseUnitless<T_Params>
                        , public incidentField::detail::BaseSeparableTransversalGaussianFunctorE<T_Params>
                    {
                        //! Unitless parameters type
                        using Unitless = ExpRampWithPrepulseUnitless<T_Params>;

                        //! Base functor type
                        using Base = incidentField::detail::BaseSeparableTransversalGaussianFunctorE<T_Params>;

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE ExpRampWithPrepulseFunctorIncidentE(
                            float_X const currentStep,
                            float3_64 const unitField)
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
                            return Base::operator()(*this, totalCellIdx);
                        }

                        /** Get time-dependent longitudinal scalar factor for the given time
                         *
                         * Interface required by Base.
                         * Gaussian transversal profile not implemented in this class, but provided by Base.
                         *
                         * @param time time moment to calculate the factor at
                         * @param phaseShift additional phase shift to add on top of everything else,
                         *                   in radian
                         */
                        HDINLINE float_X getLongitudinal(float_X const time, float_X const phaseShift) const
                        {
                            auto const phase = Unitless::w * time + Unitless::LASER_PHASE + phaseShift;
                            return math::sin(phase) * getEnvelope(time);
                        }

                    private:
                        HDINLINE float_X getEnvelope(float_X const runTime) const
                        {
                            constexpr auto int_ratio_prepule = Unitless::INT_RATIO_PREPULSE;
                            constexpr auto int_ratio_point_1 = Unitless::INT_RATIO_POINT_1;
                            constexpr auto int_ratio_point_2 = Unitless::INT_RATIO_POINT_2;
                            constexpr auto int_ratio_point_3 = Unitless::INT_RATIO_POINT_3;
                            auto const AMP_PREPULSE = math::sqrt(int_ratio_prepule) * Unitless::AMPLITUDE;
                            auto const AMP_1 = math::sqrt(int_ratio_point_1) * Unitless::AMPLITUDE;
                            auto const AMP_2 = math::sqrt(int_ratio_point_2) * Unitless::AMPLITUDE;
                            auto const AMP_3 = math::sqrt(int_ratio_point_3) * Unitless::AMPLITUDE;

                            auto env = 0.0_X;
                            bool const before_preupramp = runTime < Unitless::time_start_init;
                            bool const before_start = runTime < Unitless::TIME_1;
                            bool const before_peakpulse = runTime < Unitless::endUpramp;
                            bool const during_first_exp = (Unitless::TIME_1 < runTime) && (runTime < Unitless::TIME_2);
                            bool const after_peakpulse = Unitless::startDownramp <= runTime;

                            if(before_preupramp)
                                env = 0._X;
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

                                // This check exists in original laser, but can't print from device
                                // if(ramp_when_peakpulse > 0.5_X)
                                //{
                                //    log<picLog::PHYSICS>(
                                //        "Attention, the intensities of the laser upramp are very large! "
                                //        "The extrapolation of the last exponential to the time of "
                                //        "the peakpulse gives more than half of the amplitude of "
                                //        "the peak Gaussian. This is not a Gaussian at all anymore, "
                                //        "and physically very unplausible, check the params for misunderstandings!");
                                //}

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
                        HDINLINE float_X gauss(float_X const t) const
                        {
                            auto const exponent = t / Unitless::PULSE_LENGTH;
                            return math::exp(-0.25_X * exponent * exponent);
                        }

                        /** get value of exponential curve through two points at given t
                         * t/t1/t2 given as float_X, since the envelope doesn't need the accuracy
                         */
                        HDINLINE float_X extrapolateExpo(
                            float_X const t1,
                            float_X const a1,
                            float_X const t2,
                            float_X const a2,
                            float_X const t) const
                        {
                            auto const log1 = (t2 - t) * math::log(a1);
                            auto const log2 = (t - t1) * math::log(a2);
                            return math::exp((log1 + log2) / (t2 - t1));
                        }
                    };
                } // namespace detail
            } // namespace profiles

            namespace detail
            {
                /** Get type of incident field E functor for the exponential ramp with prepulse profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentE<profiles::ExpRampWithPrepulse<T_Params>>
                {
                    using type = profiles::detail::ExpRampWithPrepulseFunctorIncidentE<T_Params>;
                };

                /** Get type of incident field B functor for the exponential ramp with prepulse  profile type
                 *
                 * Rely on SVEA to calculate value of B from E.
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentB<profiles::ExpRampWithPrepulse<T_Params>>
                {
                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::ExpRampWithPrepulse<T_Params>>::type>;
                };
            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
