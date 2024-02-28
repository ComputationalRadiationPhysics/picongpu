/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Stefan Tietze, Sergei Bastrakov
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


namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace profiles
            {
                template<typename T_Params>
                struct Wavepacket
                {
                    //! Get text name of the incident field profile
                    HINLINE static std::string getName()
                    {
                        return "Wavepacket";
                    }
                };

                namespace detail
                {
                    /** Unitless wavepacket parameters
                     *
                     * Only parameters to be used for calculations inside kernel are float_X.
                     * For others we can use float_64 basically without any overhead.
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct WavepacketUnitless : public BaseTransversalGaussianParamUnitless<T_Params>
                    {
                        //! User SI parameters
                        using Params = T_Params;

                        //! Base unitless parameters
                        using Base = BaseTransversalGaussianParamUnitless<T_Params>;

                        // unit: UNIT_TIME
                        static constexpr float_X LASER_NOFOCUS_CONSTANT
                            = static_cast<float_X>(Params::LASER_NOFOCUS_CONSTANT_SI / UNIT_TIME);

                        // unit: UNIT_TIME
                        static constexpr float_X INIT_TIME
                            = Params::PULSE_INIT * Base::PULSE_DURATION + LASER_NOFOCUS_CONSTANT;
                        // unit: UNIT_TIME
                        static constexpr float_X endUpramp = -0.5_X * LASER_NOFOCUS_CONSTANT;
                        // unit: UNIT_TIME
                        static constexpr float_X startDownramp = 0.5_X * LASER_NOFOCUS_CONSTANT;
                    };

                    /** Wavepacket incident E functor
                     *
                     * @tparam T_Params parameters
                     */
                    template<typename T_Params>
                    struct WavepacketFunctorIncidentE
                        : public WavepacketUnitless<T_Params>
                        , public incidentField::detail::BaseSeparableTransversalGaussianFunctorE<T_Params>
                    {
                        //! Unitless parameters type
                        using Unitless = WavepacketUnitless<T_Params>;

                        //! Base functor type
                        using Base = incidentField::detail::BaseSeparableTransversalGaussianFunctorE<T_Params>;

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE WavepacketFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
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
                            // a symmetric pulse will be initialized at generation position
                            // a time of PULSE_INIT * PULSE_DURATION + LASER_NOFOCUS_CONSTANT = INIT_TIME.
                            // we shift the complete pulse for the half of this time to start with
                            // the front of the laser pulse.
                            auto const mue = 0.5_X * Unitless::INIT_TIME;
                            auto const runTime = static_cast<float_X>(time) - mue;
                            auto const tau = Unitless::PULSE_DURATION * math::sqrt(2.0_X);
                            auto envelope = Unitless::AMPLITUDE;
                            auto correctionFactor = 0.0_X;
                            if(runTime > Unitless::startDownramp)
                            {
                                // downramp = end
                                auto const exponent
                                    = ((runTime - Unitless::startDownramp) / Unitless::PULSE_DURATION
                                       / math::sqrt(2.0_X));
                                envelope *= math::exp(-0.5_X * exponent * exponent);
                                correctionFactor = (runTime - Unitless::startDownramp) / (tau * tau * Unitless::w);
                            }
                            else if(runTime < Unitless::endUpramp)
                            {
                                // upramp = start
                                auto const exponent
                                    = ((runTime - Unitless::endUpramp) / Unitless::PULSE_DURATION / math::sqrt(2.0_X));
                                envelope *= math::exp(-0.5_X * exponent * exponent);
                                correctionFactor = (runTime - Unitless::endUpramp) / (tau * tau * Unitless::w);
                            }
                            auto const phase = Unitless::w * runTime + Unitless::LASER_PHASE + phaseShift;
                            return (math::sin(phase) + correctionFactor * math::cos(phase)) * envelope;
                        }
                    };
                } // namespace detail
            } // namespace profiles

            namespace detail
            {
                /** Get type of incident field E functor for the wavepacket profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentE<profiles::Wavepacket<T_Params>>
                {
                    using type = profiles::detail::WavepacketFunctorIncidentE<T_Params>;
                };

                /** Get type of incident field B functor for the wavepacket profile type
                 *
                 * Rely on SVEA to calculate value of B from E.
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentB<profiles::Wavepacket<T_Params>>
                {
                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::Wavepacket<T_Params>>::type>;
                };
            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
