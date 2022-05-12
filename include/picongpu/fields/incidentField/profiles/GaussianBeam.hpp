/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Anton Helm, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Sergei Bastrakov
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
#include <limits>
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
                    /** Base class providing tilt value based on given parameters
                     *
                     * General implementation sets tilt to 0 and does not require T_Params having tilt as member.
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params, typename T_Sfinae = void>
                    struct TiltParam
                    {
                        static constexpr float_X TILT_AXIS_2 = 0.0_X; // unit: radiant (in dimensions of pi)
                    };

                    /** Helper type to check if T_Params has member TILT_AXIS_2_SI
                     *
                     * Is void for those types, ill-formed otherwise.
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    using HasTilt = std::void_t<decltype(T_Params::TILT_AXIS_2_SI)>;

                    /** Specialization for T_Params having tilt as member, then use it.
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct TiltParam<T_Params, HasTilt<T_Params>>
                    {
                        static constexpr float_X TILT_AXIS_2 = static_cast<float_X>(
                            T_Params::TILT_AXIS_2_SI * PI / 180.); // unit: radiant (in dimensions of pi)
                    };

                    /** Unitless gaussian beam parameters
                     *
                     * These parameters are shared for tilted and non-tilted Gaussian laser.
                     * The branching in terms of if and how user sets a tilt is encapculated in TiltParam.
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct GaussianBeamUnitless
                        : public T_Params
                        , public TiltParam<T_Params>
                    {
                        using Params = T_Params;

                        static constexpr float_X WAVE_LENGTH
                            = static_cast<float_X>(Params::WAVE_LENGTH_SI / UNIT_LENGTH); // unit: meter
                        static constexpr float_X PULSE_LENGTH
                            = static_cast<float_X>(Params::PULSE_LENGTH_SI / UNIT_TIME); // unit: seconds (1 sigma)
                        static constexpr float_X AMPLITUDE
                            = static_cast<float_X>(Params::AMPLITUDE_SI / UNIT_EFIELD); // unit: Volt /meter
                        static constexpr float_X W0 = float_X(Params::W0_SI / UNIT_LENGTH); // unit: meter
                        // rayleigh length in propagation direction
                        static constexpr float_X R = static_cast<float_X>(PI) * W0 * W0 / WAVE_LENGTH;
                        static constexpr float_X FOCUS_POS
                            = static_cast<float_X>(Params::FOCUS_POS_SI / UNIT_LENGTH); // unit: meter
                        static constexpr float_X INIT_TIME = static_cast<float_X>(
                            (Params::PULSE_INIT * Params::PULSE_LENGTH_SI)
                            / UNIT_TIME); // unit: seconds (full initialization length)

                        static constexpr float_X f = static_cast<float_X>(SPEED_OF_LIGHT / WAVE_LENGTH);
                    };

                    /** Gaussian beam incident E functor
                     *
                     * The implementation is shared between a normal Gaussian beam and one with tilted front.
                     * We always take tilt value from the unitless params and apply the tile (which can be 0).
                     *
                     * @tparam T_Params parameters
                     * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                     * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from
                     * the max boundary inwards)
                     */
                    template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                    struct GaussianBeamFunctorIncidentE
                        : public GaussianBeamUnitless<T_Params>
                        , public BaseFunctorE<T_axis, T_direction>
                    {
                        //! Unitless parameters type
                        using Unitless = GaussianBeamUnitless<T_Params>;

                        //! Base functor type
                        using Base = BaseFunctorE<T_axis, T_direction>;

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE GaussianBeamFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
                            : Base(unitField)
                            , runTime(DELTA_T * currentStep)
                        {
                            auto const& subGrid = Environment<simDim>::get().SubGrid();
                            totalDomainCells = precisionCast<float_X>(subGrid.getTotalDomain().size);

                            // This check is done here on HOST, since std::numeric_limits<float_X>::epsilon() does not
                            // compile on laserTransversal(), which is on DEVICE.
                            auto etrans_norm = 0.0_X;
                            PMACC_CASSERT_MSG(
                                MODENUMBER_must_be_smaller_than_number_of_entries_in_LAGUERREMODES_vector,
                                Unitless::MODENUMBER < Unitless::LAGUERREMODES_t::dim);
                            for(uint32_t m = 0; m <= Unitless::MODENUMBER; ++m)
                                etrans_norm += typename Unitless::LAGUERREMODES_t{}[m];
                            PMACC_VERIFY_MSG(
                                math::abs(etrans_norm) > std::numeric_limits<float_X>::epsilon(),
                                "Sum of LAGUERREMODES can not be 0.");
                        }

                        /** Calculate incident field E value for the given position
                         *
                         * The transverse spatial laser modes are given as a decomposition of Gauss-Laguerre modes
                         * GLM(m,r,z) : Sum_{m=0}^{m_max} := Snorm * a_m * GLM(m,r,z)
                         * with a_m being complex-valued coefficients: a_m := |a_m| * exp(I Arg(a_m) )
                         * |a_m| are equivalent to the LAGUERREMODES vector entries.
                         * Arg(a_m) are equivalent to the LAGUERREPHASES vector entries.
                         * The implicit beam properties w0, lambda0, etc... equally apply to all GLM-modes.
                         * The on-axis, in-focus field value of the mode decomposition is normalized to unity:
                         * Snorm := 1 / ( Sum_{m=0}^{m_max}GLM(m,0,0) )
                         *
                         * Spatial mode: Arg(a_m) * GLM(m,r,z) := w0/w(zeta) * L_m( 2*r^2/(w(zeta))^2 ) \
                         *     * exp( I*k*z - I*(2*m+1)*ArcTan(zeta) - r^2 / ( w0^2*(1+I*zeta) ) + I*Arg(a_m) ) )
                         * with w(zeta) = w0*sqrt(1+zeta^2)
                         * with zeta = z / zR
                         * with zR = PI * w0^2 / lambda0
                         *
                         * Uses only radial modes (m) of Laguerre-Polynomials: L_m(x)=L_m^n=0(x)
                         * z is the direction of laser propagation. In PIConGPU, the direction of propagation is y.
                         *
                         * References:
                         * F. Pampaloni et al. (2004), Gaussian, Hermite-Gaussian, and Laguerre-Gaussian beams: A
                         * primer https://arxiv.org/pdf/physics/0410021
                         *
                         * Allen, L. (June 1, 1992). "Orbital angular momentum of light
                         *      and the transformation of Laguerre-Gaussian laser modes"
                         * https://doi.org/10.1103/physreva.45.8185
                         *
                         * Wikipedia on Gaussian laser beams
                         * https://en.wikipedia.org/wiki/Gaussian_beam
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @return incident field E value in internal units
                         */
                        HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
                        {
                            // transform coordinate system to center of x-z plane of initialization
                            floatD_X pos = (totalCellIdx - totalDomainCells * 0.5_X) * cellSize.shrink<simDim>();
                            if(T_direction > 0)
                                pos[Base::dir0] = totalCellIdx[Base::dir0] * cellSize[Base::dir0];
                            else
                                pos[Base::dir0]
                                    = (totalDomainCells[Base::dir0] - totalCellIdx[Base::dir0]) * cellSize[Base::dir0];
                            floatD_X planeNoNormal = floatD_X::create(1.0_X);
                            planeNoNormal[Base::dir0] = 0.0_X;

                            // calculate focus position relative to the current point in the propagation direction
                            float_X const focusPos = Unitless::FOCUS_POS - pos[Base::dir0];
                            // beam waist at the generation plane so that at focus we will get W0
                            float_X const w = Unitless::W0
                                * math::sqrt(1.0_X + (focusPos / Unitless::R) * (focusPos / Unitless::R));

                            auto result = getEnvelope(w);
                            // a symmetric pulse will be initialized at position z=0 for
                            // a time of PULSE_INIT * PULSE_LENGTH = INIT_TIME.
                            // we shift the complete pulse for the half of this time to start with
                            // the front of the laser pulse.
                            constexpr auto mue = 0.5_X * Unitless::INIT_TIME;
                            auto const phase = 2.0_X * static_cast<float_X>(PI) * Unitless::f
                                    * (runTime - mue - focusPos / SPEED_OF_LIGHT)
                                + Unitless::LASER_PHASE;

                            // Apply tilt in Base::dir2
                            auto const timeShift
                                = phase / (2.0_X * float_X(PI) * float_X(Unitless::f)) + focusPos / SPEED_OF_LIGHT;
                            auto const tilt = Unitless::TILT_AXIS_2;
                            auto const shiftAxis2
                                = SPEED_OF_LIGHT * math::tan(tilt) * timeShift / cellSize[Base::dir0];
                            pos[Base::dir2] += shiftAxis2;
                            float_X const transversalDistanceSquared = pmacc::math::abs2(pos * planeNoNormal);

                            if(Unitless::Polarisation == Unitless::LINEAR_AXIS_2
                               || Unitless::Polarisation == Unitless::LINEAR_AXIS_1)
                            {
                                result *= getValue(phase, focusPos, w, transversalDistanceSquared);
                            }
                            else if(Unitless::Polarisation == Unitless::CIRCULAR)
                            {
                                result[Base::dir2] *= getValue(phase, focusPos, w, transversalDistanceSquared);
                                result[Base::dir1]
                                    *= getValue(phase + float_X(PI / 2.0), focusPos, w, transversalDistanceSquared);
                            }
                            return result;
                        }

                    private:
                        //! Total domain size in cells
                        floatD_X totalDomainCells;

                        //! Current time
                        float_X const runTime;

                        /** Get vector field with waist-dependent envelope in components according to the polarization
                         *
                         * This function merely applies envelope and polarization, most calculations to produce a_m
                         * Gaussian pulse happen elsewhere.
                         *
                         * @param w unitless beam waist
                         */
                        HDINLINE float3_X getEnvelope(float_X const w) const
                        {
                            auto envelope = Unitless::AMPLITUDE;
                            if(simDim == DIM2)
                                envelope *= math::sqrt(Unitless::W0 / w);
                            else if(simDim == DIM3)
                                envelope *= Unitless::W0 / w;

                            auto result = float3_X::create(0.0_X);
                            if(Unitless::Polarisation == Unitless::LINEAR_AXIS_2)
                            {
                                result[Base::dir2] = envelope;
                            }
                            else if(Unitless::Polarisation == Unitless::LINEAR_AXIS_1)
                            {
                                result[Base::dir1] = envelope;
                            }
                            else if(Unitless::Polarisation == Unitless::CIRCULAR)
                            {
                                result[Base::dir1] = envelope / math::sqrt(2.0_X);
                                result[Base::dir2] = envelope / math::sqrt(2.0_X);
                            }
                            return result;
                        }

                        /** Get scalar multiplier to the envelope
                         *
                         * Does most calculations to produce a Gaussian pulse.
                         *
                         * @param phase phase value
                         * @param focusPos distance to focus position in the propagation direction
                         * @param w unitless beam waist
                         * @param transversalDistanceSquared squared distance from beam center in the transversal plane
                         */
                        HDINLINE float_X getValue(
                            float_X const phase,
                            float_X const focusPos,
                            float_X const w,
                            float_X const transversalDistanceSquared) const
                        {
                            // inverse radius of curvature of the beam's  wavefronts
                            float_X const R_inv = -focusPos / (Unitless::R * Unitless::R + focusPos * focusPos);
                            // the Gouy phase shift
                            float_X const xi = math::atan(-focusPos / Unitless::R);
                            auto etrans = 0.0_X;
                            auto const r2OverW2 = transversalDistanceSquared / w / w;
                            auto const r = 0.5_X * transversalDistanceSquared * R_inv;
                            for(uint32_t m = 0; m <= Unitless::MODENUMBER; ++m)
                            {
                                etrans += typename Unitless::LAGUERREMODES_t{}[m] * simpleLaguerre(m, 2.0_X * r2OverW2)
                                    * math::exp(-r2OverW2)
                                    * math::cos(
                                              2.0_X * float_X(PI) / Unitless::WAVE_LENGTH * focusPos
                                              - 2.0_X * float_X(PI) / Unitless::WAVE_LENGTH * r
                                              + (2._X * float_X(m) + 1._X) * xi + phase +
                                              typename Unitless::LAGUERREPHASES_t{}[m]);
                            }
                            auto const exponent = (r - focusPos - phase / 2.0_X / float_X(PI) * Unitless::WAVE_LENGTH)
                                / (SPEED_OF_LIGHT * 2.0_X * Unitless::PULSE_LENGTH);
                            etrans *= math::exp(-exponent * exponent);
                            auto etrans_norm = 0.0_X;
                            for(uint32_t m = 0; m <= Unitless::MODENUMBER; ++m)
                                etrans_norm += typename Unitless::LAGUERREMODES_t{}[m];
                            return etrans / etrans_norm;
                        }

                        /** Simple iteration algorithm to implement Laguerre polynomials for GPUs.
                         *
                         *  @param n order of the Laguerre polynomial
                         *  @param x coordinate at which the polynomial is evaluated
                         */
                        HDINLINE float_X simpleLaguerre(uint32_t const n, float_X const x) const
                        {
                            // Result for special case n == 0
                            if(n == 0)
                                return 1.0_X;
                            uint32_t currentN = 1;
                            float_X laguerreNMinus1 = 1.0_X;
                            float_X laguerreN = 1.0_X - x;
                            float_X laguerreNPlus1(0.0_X);
                            while(currentN < n)
                            {
                                // Core statement of the algorithm
                                laguerreNPlus1 = ((2.0_X * float_X(currentN) + 1.0_X - x) * laguerreN
                                                  - float_X(currentN) * laguerreNMinus1)
                                    / float_X(currentN + 1u);
                                // Advance by one order
                                laguerreNMinus1 = laguerreN;
                                laguerreN = laguerreNPlus1;
                                currentN++;
                            }
                            return laguerreN;
                        }
                    };
                } // namespace detail
            } // namespace profiles

            namespace detail
            {
                /** Get type of incident field E functor for the gaussian beam profile type
                 *
                 * @tparam T_Params parameters
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 */
                template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                struct GetFunctorIncidentE<profiles::GaussianBeam<T_Params>, T_axis, T_direction>
                {
                    using type = profiles::detail::GaussianBeamFunctorIncidentE<T_Params, T_axis, T_direction>;
                };

                /** Get type of incident field B functor for the gaussiam beam profile type
                 *
                 * Rely on SVEA to calculate value of B from E.
                 *
                 * @tparam T_Params parameters
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 */
                template<typename T_Params, uint32_t T_axis, int32_t T_direction>
                struct GetFunctorIncidentB<profiles::GaussianBeam<T_Params>, T_axis, T_direction>
                {
                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::GaussianBeam<T_Params>, T_axis, T_direction>::type>;
                };
            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
