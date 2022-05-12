/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
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
                    /** Unitless wavepacket parameters
                     *
                     * Only parameters to be used for calculations inside kernel are float_X.
                     * For others we can use float_64 basically without any overhead.
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct WavepacketUnitless : public T_Params
                    {
                        using Params = T_Params;

                        static constexpr float_64 WAVE_LENGTH = Params::WAVE_LENGTH_SI / UNIT_LENGTH; // unit: meter
                        static constexpr float_64 PULSE_LENGTH
                            = Params::PULSE_LENGTH_SI / UNIT_TIME; // unit: seconds (1 sigma)
                        static constexpr float_X LASER_NOFOCUS_CONSTANT
                            = float_X(Params::LASER_NOFOCUS_CONSTANT_SI / UNIT_TIME); // unit: seconds
                        static constexpr float_64 AMPLITUDE = Params::AMPLITUDE_SI / UNIT_EFIELD; // unit: Volt /meter
                        static constexpr float_X W0_AXIS_1
                            = float_X(Params::W0_AXIS_1_SI / UNIT_LENGTH); // unit: meter
                        static constexpr float_X W0_AXIS_2
                            = float_X(Params::W0_AXIS_2_SI / UNIT_LENGTH); // unit: meter
                        static constexpr float_64 INIT_TIME = Params::PULSE_INIT * PULSE_LENGTH
                            + LASER_NOFOCUS_CONSTANT; // unit: seconds (full initialization length)
                        static constexpr float_64 endUpramp = -0.5_X * LASER_NOFOCUS_CONSTANT; // unit: seconds
                        static constexpr float_64 startDownramp = 0.5_X * LASER_NOFOCUS_CONSTANT; // unit: seconds
                        static constexpr float_64 f = SPEED_OF_LIGHT / WAVE_LENGTH;
                        static constexpr float_64 w = 2.0 * PI * f;
                    };

                    /** Wavepacket incident E functor
                     *
                     * @tparam T_Params parameters
                     * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                     * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from
                     * the max boundary inwards)
                     */
                    template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                    struct WavepacketFunctorIncidentE
                        : public WavepacketUnitless<T_Params>
                        , public BaseFunctorE<T_axis, T_direction>
                    {
                        //! Unitless parameters type
                        using Unitless = WavepacketUnitless<T_Params>;

                        //! Base functor type
                        using Base = BaseFunctorE<T_axis, T_direction>;

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE WavepacketFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
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

                        //! Get time-dependent longitudinal vector field
                        HDINLINE float3_X getLongitudinal(float_X const currentStep) const
                        {
                            // a symmetric pulse will be initialized at position z=0 for
                            // a time of PULSE_INIT * PULSE_LENGTH + LASER_NOFOCUS_CONSTANT = INIT_TIME.
                            // we shift the complete pulse for the half of this time to start with
                            // the front of the laser pulse.
                            auto const mue = 0.5 * Unitless::INIT_TIME;
                            auto const runTime = static_cast<float_64>(DELTA_T * currentStep) - mue;
                            auto const tau = Unitless::PULSE_LENGTH * math::sqrt(2.0);
                            auto envelope = Unitless::AMPLITUDE;
                            auto correctionFactor = 0.0;
                            if(runTime > Unitless::startDownramp)
                            {
                                // downramp = end
                                auto const exponent
                                    = ((runTime - Unitless::startDownramp) / Unitless::PULSE_LENGTH / math::sqrt(2.0));
                                envelope *= math::exp(-0.5 * exponent * exponent);
                                correctionFactor = (runTime - Unitless::startDownramp) / (tau * tau * Unitless::w);
                            }
                            else if(runTime < Unitless::endUpramp)
                            {
                                // upramp = start
                                auto const exponent
                                    = ((runTime - Unitless::endUpramp) / Unitless::PULSE_LENGTH / math::sqrt(2.0_X));
                                envelope *= math::exp(-0.5 * exponent * exponent);
                                correctionFactor = (runTime - Unitless::endUpramp) / (tau * tau * Unitless::w);
                            }

                            auto result = float3_X::create(0.0_X);
                            auto const phase = Unitless::w * runTime + Unitless::LASER_PHASE;
                            auto const baseValue = math::sin(phase) + correctionFactor * math::cos(phase);
                            if(Unitless::Polarisation == Unitless::LINEAR_AXIS_2)
                                result[Base::dir2] = baseValue;
                            else if(Unitless::Polarisation == Unitless::LINEAR_AXIS_1)
                            {
                                result[Base::dir1] = baseValue;
                            }
                            else if(Unitless::Polarisation == Unitless::CIRCULAR)
                            {
                                result[Base::dir2] = baseValue / math::sqrt(2.0);
                                result[Base::dir1]
                                    = (math::cos(phase) + correctionFactor * math::sin(phase)) / math::sqrt(2.0);
                            }
                            return result * static_cast<float_X>(envelope);
                        }

                        //! Get position-dependent transversal scalar multiplier
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
                /** Get type of incident field E functor for the wavepacket profile type
                 *
                 * @tparam T_Params parameters
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 */
                template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                struct GetFunctorIncidentE<profiles::Wavepacket<T_Params>, T_axis, T_direction>
                {
                    using type = profiles::detail::WavepacketFunctorIncidentE<T_Params, T_axis, T_direction>;
                };

                /** Get type of incident field B functor for the wavepacket profile type
                 *
                 * Rely on SVEA to calculate value of B from E.
                 *
                 * @tparam T_Params parameters
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 */
                template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                struct GetFunctorIncidentB<profiles::Wavepacket<T_Params>, T_axis, T_direction>
                {
                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::Wavepacket<T_Params>, T_axis, T_direction>::type>;
                };
            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
