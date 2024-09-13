/* Copyright 2013-2023 Heiko Burau, Rene Widera, Richard Pausch, Axel Huebl, Sergei Bastrakov
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
                struct Polynom
                {
                    //! Get text name of the incident field profile
                    HINLINE static std::string getName()
                    {
                        return "Polynom";
                    }
                };

                namespace detail
                {
                    /** Unitless polynom parameters
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct PolynomUnitless : public BaseTransversalGaussianParamUnitless<T_Params>
                    {
                        //! User SI parameters
                        using Params = T_Params;

                        //! Base unitless parameters
                        using Base = BaseTransversalGaussianParamUnitless<T_Params>;

                        // unit: sim.unit.time()
                        static constexpr float_X INIT_TIME
                            = static_cast<float_X>(Params::PULSE_DURATION_SI / sim.unit.time());
                    };

                    /** Polynom incident E functor
                     *
                     * @tparam T_Params parameters
                     */
                    template<typename T_Params>
                    struct PolynomFunctorIncidentE
                        : public PolynomUnitless<T_Params>
                        , public incidentField::detail::BaseSeparableTransversalGaussianFunctorE<T_Params>
                    {
                        //! Unitless parameters type
                        using Unitless = PolynomUnitless<T_Params>;

                        //! Base functor type
                        using Base = incidentField::detail::BaseSeparableTransversalGaussianFunctorE<T_Params>;

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE PolynomFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
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
                            /* a symmetric pulse will be initialized at generation position
                             * the laser amplitude rises  for riseTime and falls for riseTime
                             * making the laser pulse 2*riseTime long
                             */
                            const float_X riseTime = 0.5_X * Unitless::PULSE_DURATION;
                            const float_X tau = time / riseTime;
                            auto const phase = Unitless::w * (time - riseTime) + Unitless::LASER_PHASE + phaseShift;
                            auto const amplitude = Unitless::AMPLITUDE * polynomial(tau);
                            return math::sin(phase) * amplitude;
                        }

                    private:
                        //! Get polynomial value for unitless time variable tau in [0.0, 2.0]
                        HDINLINE static float_X polynomial(float_X const tau)
                        {
                            auto result = 0.0_X;
                            if(tau >= 0.0_X && tau <= 1.0_X)
                                result = tau * tau * tau * (10.0_X - 15.0_X * tau + 6.0_X * tau * tau);
                            else if(tau > 1.0_X && tau <= 2.0_X)
                                result = (2.0_X - tau) * (2.0_X - tau) * (2.0_X - tau)
                                    * (4.0_X - 9.0_X * tau + 6.0_X * tau * tau);
                            return result;
                        }
                    };
                } // namespace detail
            } // namespace profiles

            namespace traits::detail
            {
                /** Get type of incident field E functor for the polynom profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentE<profiles::Polynom<T_Params>>
                {
                    using type = profiles::detail::PolynomFunctorIncidentE<T_Params>;
                };

                /** Get type of incident field B functor for the polynom profile type
                 *
                 * Rely on SVEA to calculate value of B from E.
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentB<profiles::Polynom<T_Params>>
                {
                    using type = incidentField::detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::Polynom<T_Params>>::type>;
                };
            } // namespace traits::detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
