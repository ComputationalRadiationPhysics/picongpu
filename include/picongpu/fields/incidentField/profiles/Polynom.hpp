/* Copyright 2013-2022 Heiko Burau, Rene Widera, Richard Pausch, Axel Huebl, Sergei Bastrakov
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
                    /** Unitless polynom parameters
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct PolynomUnitless : public T_Params
                    {
                        using Params = T_Params;

                        static constexpr float_X WAVE_LENGTH
                            = float_X(Params::WAVE_LENGTH_SI / UNIT_LENGTH); // unit: meter
                        static constexpr float_X PULSE_LENGTH
                            = float_X(Params::PULSE_LENGTH_SI / UNIT_TIME); // unit: seconds (1 sigma)
                        static constexpr float_X AMPLITUDE
                            = float_X(Params::AMPLITUDE_SI / UNIT_EFIELD); // unit: Volt /meter
                        static constexpr float_X W0_AXIS_1
                            = float_X(Params::W0_AXIS_1_SI / UNIT_LENGTH); // unit: meter
                        static constexpr float_X W0_AXIS_2
                            = float_X(Params::W0_AXIS_2_SI / UNIT_LENGTH); // unit: meter
                        static constexpr float_X INIT_TIME = float_X(
                            Params::PULSE_LENGTH_SI / UNIT_TIME); // unit: seconds (full initialization length)
                        static constexpr float_64 f = SPEED_OF_LIGHT / WAVE_LENGTH;
                    };

                    /** Polynom incident E functor
                     *
                     * @tparam T_Params parameters
                     * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                     * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from
                     * the max boundary inwards)
                     */
                    template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                    struct PolynomFunctorIncidentE
                        : public PolynomUnitless<T_Params>
                        , public BaseFunctorE<T_axis, T_direction>
                    {
                        //! Unitless parameters type
                        using Unitless = PolynomUnitless<T_Params>;

                        //! Base functor type
                        using Base = BaseFunctorE<T_axis, T_direction>;

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE PolynomFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
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
                        HDINLINE float3_X getLongitudinal(float_X const currentStep) const
                        {
                            auto result = float3_X::create(0.0_X);
                            /* a symmetric pulse will be initialized at position z=0
                             * the laser amplitude rises  for riseTime and falls for riseTime
                             * making the laser pulse 2*riseTime long
                             */
                            float_64 const runTime = DELTA_T * currentStep;
                            const float_X riseTime = 0.5_X * Unitless::PULSE_LENGTH;
                            const float_X tau = runTime / riseTime;
                            const float_X omegaLaser = 2.0_X * PI * Unitless::f;
                            auto const arg = omegaLaser * (runTime - riseTime) + Unitless::LASER_PHASE;
                            if(Unitless::Polarisation == Unitless::LINEAR_AXIS_2)
                                result[Base::dir2] = math::sin(arg);
                            else if(Unitless::Polarisation == Unitless::LINEAR_AXIS_1)
                                result[Base::dir1] = math::sin(arg);
                            else if(Unitless::Polarisation == Unitless::CIRCULAR)
                            {
                                result[Base::dir2] = math::sin(arg) / math::sqrt(2.0_X);
                                result[Base::dir1] = math::cos(arg) / math::sqrt(2.0_X);
                            }
                            auto const amplitude = static_cast<float_X>(Unitless::AMPLITUDE * polynomial(tau));
                            return result * amplitude;
                        }

                        //! Get polynomial value for unitless time variable tau in [0.0, 2.0]
                        static HDINLINE float_X polynomial(float_X const tau)
                        {
                            auto result = 0.0_X;
                            if(tau >= 0.0_X && tau <= 1.0_X)
                                result = tau * tau * tau * (10.0_X - 15.0_X * tau + 6.0_X * tau * tau);
                            else if(tau > 1.0_X && tau <= 2.0_X)
                                result = (2.0_X - tau) * (2.0_X - tau) * (2.0_X - tau)
                                    * (4.0_X - 9.0_X * tau + 6.0_X * tau * tau);
                            return result;
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
                /** Get type of incident field E functor for the polynom profile type
                 *
                 * @tparam T_Params parameters
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 */
                template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                struct GetFunctorIncidentE<profiles::Polynom<T_Params>, T_axis, T_direction>
                {
                    using type = profiles::detail::PolynomFunctorIncidentE<T_Params, T_axis, T_direction>;
                };

                /** Get type of incident field B functor for the polynom profile type
                 *
                 * Rely on SVEA to calculate value of B from E.
                 *
                 * @tparam T_Params parameters
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 */
                template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                struct GetFunctorIncidentB<profiles::Polynom<T_Params>, T_axis, T_direction>
                {
                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::Polynom<T_Params>, T_axis, T_direction>::type>;
                };
            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
