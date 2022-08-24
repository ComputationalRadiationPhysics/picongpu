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
                    /** Base class providing tilt value based on given parameters
                     *
                     * General implementation sets tilts to 0 and does not require T_Params having tilt as member.
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params, typename T_Sfinae = void>
                    struct TiltParam
                    {
                        // unit: radian
                        static constexpr float_X TILT_AXIS_1 = 0.0_X;
                        // unit: radian
                        static constexpr float_X TILT_AXIS_2 = 0.0_X;
                    };

                    /** Helper type to check if T_Params has members TILT_AXIS_1_SI and TILT_AXIS_2_SI.
                     *
                     * Is void for those types, ill-formed otherwise.
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    using HasTilt = std::void_t<decltype(T_Params::TILT_AXIS_1_SI + T_Params::TILT_AXIS_2_SI)>;

                    /** Specialization for T_Params having tilt as member, then use it.
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct TiltParam<T_Params, HasTilt<T_Params>>
                    {
                        // unit: radian
                        static constexpr float_X TILT_AXIS_1
                            = static_cast<float_X>(T_Params::TILT_AXIS_1_SI * pmacc::math::Pi<float_X>::value / 180.);
                        // unit: radian
                        static constexpr float_X TILT_AXIS_2
                            = static_cast<float_X>(T_Params::TILT_AXIS_2_SI * pmacc::math::Pi<float_X>::value / 180.);
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
                        : public BaseParamUnitless<T_Params>
                        , public TiltParam<T_Params>
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
                        static constexpr float_X INIT_TIME
                            = static_cast<float_X>((Params::PULSE_INIT * Params::PULSE_LENGTH_SI) / UNIT_TIME);
                    };

                    /** Gaussian beam incident E functor
                     *
                     * The implementation is shared between a normal Gaussian beam and one with tilted front.
                     * We always take tilt value from the unitless params and apply the tilt (which can be 0).
                     *
                     * @tparam T_Params parameters
                     */
                    template<typename T_Params>
                    struct GaussianBeamFunctorIncidentE
                        : public GaussianBeamUnitless<T_Params>
                        , public incidentField::detail::BaseFunctorE<T_Params>
                    {
                        //! Unitless parameters type
                        using Unitless = GaussianBeamUnitless<T_Params>;

                        //! Base functor type
                        using Base = incidentField::detail::BaseFunctorE<T_Params>;

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE GaussianBeamFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
                            : Base(currentStep, unitField)
                        {
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
                         * In the formula above, z is the direction of laser propagation. 
                         * In PIConGPU, the propagation direction can be chosen freely. In the following code,
                         * pos[0] is the propagation direction.
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
                            if(Unitless::Polarisation == PolarisationType::Linear)
                                return this->getLinearPolarizationVector() * getValue(totalCellIdx, 0.0_X);
                            else
                            {
                                auto const phaseShift = pmacc::math::Pi<float_X>::halfValue;
                                return this->getCircularPolarizationVector1() * getValue(totalCellIdx, phaseShift)
                                    + this->getCircularPolarizationVector2() * getValue(totalCellIdx, 0.0_X);
                            }
                        }

                    private:
                        /** Get value for the given position
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @param phaseShift additional phase shift to add on top of everything else,
                         *                   in radian
                         */
                        HDINLINE float_X getValue(floatD_X const& totalCellIdx, float_X const phaseShift) const
                        {
                            // transform to 3d internal coordinate system
                            float3_X pos = this->getInternalCoordinates(totalCellIdx);
                            auto const time = this->getCurrentTime(totalCellIdx);
                            if(time < 0.0_X)
                                return 0.0_X;

                            // calculate focus position relative to the current point in the propagation direction
                            auto const focusRelativeToOrigin = float3_X(
                                                                   Unitless::FOCUS_POSITION_X,
                                                                   Unitless::FOCUS_POSITION_Y,
                                                                   Unitless::FOCUS_POSITION_Z)
                                - this->origin;
                            float_X const focusPos = math::sqrt(pmacc::math::abs2(focusRelativeToOrigin)) - pos[0];
                            // beam waist at the generation plane so that at focus we will get W0
                            float_X const w = Unitless::W0
                                * math::sqrt(1.0_X + (focusPos / Unitless::R) * (focusPos / Unitless::R));

                            // a symmetric pulse will be initialized at generation plane for
                            // a time of PULSE_INIT * PULSE_LENGTH = INIT_TIME.
                            // we shift the complete pulse for the half of this time to start with
                            // the front of the laser pulse.
                            constexpr auto mue = 0.5_X * Unitless::INIT_TIME;
                            auto const phase = Unitless::w * (time - mue - focusPos / SPEED_OF_LIGHT)
                                + Unitless::LASER_PHASE + phaseShift;

                            // Apply tilt if needed
                            if constexpr(Unitless::TILT_AXIS_1 || Unitless::TILT_AXIS_2)
                            {
                                auto const tiltTimeShift = phase / Unitless::w + focusPos / SPEED_OF_LIGHT;
                                auto const tiltPositionShift = SPEED_OF_LIGHT * tiltTimeShift
                                    / pmacc::math::dot(this->getDirection(), float3_X{cellSize});
                                auto const tilt1 = Unitless::TILT_AXIS_1;
                                pos[1] += math::tan(tilt1) * tiltPositionShift;
                                auto const tilt2 = Unitless::TILT_AXIS_2;
                                pos[2] += math::tan(tilt2) * tiltPositionShift;
                            }

                            auto planeNoNormal = float3_X::create(1.0_X);
                            planeNoNormal[0] = 0.0_X;
                            auto const transversalDistanceSquared = pmacc::math::abs2(pos * planeNoNormal);

                            // inverse radius of curvature of the beam's  wavefronts
                            auto const R_inv = -focusPos / (Unitless::R * Unitless::R + focusPos * focusPos);
                            // the Gouy phase shift
                            auto xi = math::atan(-focusPos / Unitless::R);
                            if(simDim == DIM2)
                                xi *= 0.5_X;
                            auto etrans = 0.0_X;
                            auto const r2OverW2 = transversalDistanceSquared / w / w;
                            auto const r = 0.5_X * transversalDistanceSquared * R_inv;
                            for(uint32_t m = 0; m <= Unitless::MODENUMBER; ++m)
                            {
                                etrans += typename Unitless::LAGUERREMODES_t{}[m] * simpleLaguerre(m, 2.0_X * r2OverW2)
                                    * math::exp(-r2OverW2)
                                    * math::cos(
                                              pmacc::math::Pi<float_X>::doubleValue / Unitless::WAVE_LENGTH * focusPos
                                              - pmacc::math::Pi<float_X>::doubleValue / Unitless::WAVE_LENGTH * r
                                              + (2._X * float_X(m) + 1._X) * xi + phase +
                                              typename Unitless::LAGUERREPHASES_t{}[m]);
                            }
                            auto const exponent
                                = (r - focusPos
                                   - phase / pmacc::math::Pi<float_X>::doubleValue * Unitless::WAVE_LENGTH)
                                / (SPEED_OF_LIGHT * 2.0_X * Unitless::PULSE_LENGTH);
                            etrans *= math::exp(-exponent * exponent);
                            auto etrans_norm = 0.0_X;
                            for(uint32_t m = 0; m <= Unitless::MODENUMBER; ++m)
                                etrans_norm += typename Unitless::LAGUERREMODES_t{}[m];
                            auto envelope = Unitless::AMPLITUDE;
                            if(simDim == DIM2)
                                envelope *= math::sqrt(Unitless::W0 / w);
                            else if(simDim == DIM3)
                                envelope *= Unitless::W0 / w;
                            return envelope * etrans / etrans_norm;
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

                template<typename T_Params>
                struct GaussianBeam
                {
                    //! Get text name of the incident field profile
                    static HINLINE std::string getName()
                    {
                        // This template is used for both Gaussian and PulseFrontTilt, distinguish based on tilt value
                        using TiltParam = detail::TiltParam<T_Params>;
                        bool isTilted = (std::abs(TiltParam::TILT_AXIS_1) + std::abs(TiltParam::TILT_AXIS_2) > 0);
                        return isTilted ? "PulseFrontTilt" : "GaussianBeam";
                    }
                };

            } // namespace profiles

            namespace detail
            {
                /** Get type of incident field E functor for the gaussian beam profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentE<profiles::GaussianBeam<T_Params>>
                {
                    using type = profiles::detail::GaussianBeamFunctorIncidentE<T_Params>;
                };

                /** Get type of incident field B functor for the gaussiam beam profile type
                 *
                 * Rely on SVEA to calculate value of B from E.
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentB<profiles::GaussianBeam<T_Params>>
                {
                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::GaussianBeam<T_Params>>::type>;
                };
            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
