/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch, Sergei Bastrakov
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
                struct PlaneWave
                {
                    //! Get text name of the incident field profile
                    HINLINE static std::string getName()
                    {
                        return "PlaneWave";
                    }
                };

                namespace detail
                {
                    /** Unitless plane wave parameters
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct PlaneWaveUnitless : public BaseParamUnitless<T_Params>
                    {
                        //! User SI parameters
                        using Params = T_Params;

                        //! Base unitless parameters
                        using Base = BaseParamUnitless<T_Params>;

                        // unit: UNIT_TIME
                        static constexpr float_X LASER_NOFOCUS_CONSTANT
                            = static_cast<float_X>(Params::LASER_NOFOCUS_CONSTANT_SI / UNIT_TIME);
                        // unit: UNIT_TIME
                        static constexpr float_X INIT_TIME = static_cast<float_X>(
                            (Params::RAMP_INIT * Params::PULSE_DURATION_SI + Params::LASER_NOFOCUS_CONSTANT_SI)
                            / UNIT_TIME);
                    };

                    /** Plane wave incident E functor
                     *
                     * @tparam T_Params parameters
                     */
                    template<typename T_Params>
                    struct PlaneWaveFunctorIncidentE
                        : public PlaneWaveUnitless<T_Params>
                        , public incidentField::detail::BaseSeparableFunctorE<T_Params>
                    {
                    public:
                        //! Unitless parameters type
                        using Unitless = PlaneWaveUnitless<T_Params>;

                        //! Base functor type
                        using Base = incidentField::detail::BaseSeparableFunctorE<T_Params>;

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE PlaneWaveFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
                            : Base(currentStep, unitField)
                        {
                        }

                        /** Calculate incident field E value for the given position
                         *
                         * Interface required by Base.
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
                         * @param time time moment to calculate the factor at
                         * @param phaseShift additional phase shift to add on top of everything else,
                         *                   in radian
                         */
                        HDINLINE float_X getLongitudinal(float_X const time, float_X const phaseShift) const
                        {
                            auto envelope = Unitless::AMPLITUDE;
                            auto const mue = 0.5_X * Unitless::RAMP_INIT * Unitless::PULSE_DURATION;
                            auto const tau = Unitless::PULSE_DURATION * math::sqrt(2.0_X);
                            auto const endUpramp = mue;
                            auto const startDownramp = mue + Unitless::LASER_NOFOCUS_CONSTANT;
                            auto integrationCorrectionFactor = 0.0_X;
                            if(time > startDownramp)
                            {
                                // downramp = end
                                auto const exponent = (time - startDownramp) / tau;
                                envelope *= math::exp(-0.5_X * exponent * exponent);
                                integrationCorrectionFactor = (time - startDownramp) / (Unitless::w * tau * tau);
                            }
                            else if(time < endUpramp)
                            {
                                // upramp = start
                                auto const exponent = (time - endUpramp) / tau;
                                envelope *= math::exp(-0.5_X * exponent * exponent);
                                integrationCorrectionFactor = (time - endUpramp) / (Unitless::w * tau * tau);
                            }

                            auto const timeOszi = time - endUpramp;
                            auto const phase = Unitless::w * timeOszi + Unitless::LASER_PHASE + phaseShift;
                            // to understand both components [sin(...) + t/tau^2 * cos(...)] see description above
                            return (math::sin(phase) + math::cos(phase) * integrationCorrectionFactor) * envelope;
                        }

                        /** Get position-dependent transversal scalar factor for the given position
                         *
                         * Interface required by Base.
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         */
                        HDINLINE float_X getTransversal(floatD_X const& totalCellIdx) const
                        {
                            return 1.0_X;
                        }
                    };
                } // namespace detail
            } // namespace profiles

            namespace detail
            {
                /** Get type of incident field E functor for the plane wave profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentE<profiles::PlaneWave<T_Params>>
                {
                    using type = profiles::detail::PlaneWaveFunctorIncidentE<T_Params>;
                };

                /** Get type of incident field B functor for the plane wave profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentB<profiles::PlaneWave<T_Params>>
                {
                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::PlaneWave<T_Params>>::type>;
                };
            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
