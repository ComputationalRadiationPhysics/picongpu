/* Copyright 2013-2025 Axel Huebl, Heiko Burau, Anton Helm, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Sergei Bastrakov,
 *                     Julian Lenz, Klaus Steiniger, Pawel Ordyna
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

#include "picongpu/defines.hpp"
#include "picongpu/fields/incidentField/Functors.hpp"
#include "picongpu/fields/incidentField/profiles/BaseParam.def"

#include <pmacc/algorithms/math/defines/pi.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace picongpu::fields::incidentField
{
    namespace profiles
    {
        namespace detail
        {
            /** Value of GaussianPulse's waist sizes
             * Defined here to ensure compatibility to old setups defining only W0_SI.
             * This struct is instantiated in the default case where W0_AXIS_1_SI exists in T_Params.
             * Then, W0_AXIS_2_SI should also exist, since the params already follow the updated rules.
             *
             * @tparam T_Params user (SI) parameters
             */
            template<typename T_Params, typename T_Sfinae = void>
            struct WaistParamUnitless
            {
                static constexpr float_64 W0_AXIS_1 = static_cast<float_X>(T_Params::W0_AXIS_1_SI / sim.unit.length());
                static constexpr float_64 W0_AXIS_2 = static_cast<float_X>(T_Params::W0_AXIS_2_SI / sim.unit.length());
            };

            /** Helper type to check if T_Params has member W0_SI.
             *
             * Is void for those types, ill-formed otherwise.
             *
             * @tparam T_Params user (SI) parameters
             */
            template<typename T_Params>
            using HasW0_SI = std::void_t<decltype(T_Params::W0_SI)>;

            /** Specialization for T_Params having W0_SI as member, then use it.
             *
             * @tparam T_Params user (SI) parameters
             */
            template<typename T_Params>
            struct WaistParamUnitless<T_Params, HasW0_SI<T_Params>>
            {
                static constexpr float_64 W0_AXIS_1 = static_cast<float_X>(T_Params::W0_SI / sim.unit.length());
                static constexpr float_64 W0_AXIS_2 = W0_AXIS_1;
            };

            /** Unitless GaussianPulse parameters
             *
             * @tparam T_Params user (SI) parameters
             */
            template<typename T_Params>
            struct GaussianPulseUnitless
                : public BaseParamUnitless<T_Params>
                , public WaistParamUnitless<T_Params>
            {
                //! User SI parameters
                using Params = T_Params;

                //! Base unitless parameters
                using Base = BaseParamUnitless<T_Params>;

                //! Waist unitless parameters
                using Waist = WaistParamUnitless<T_Params>;

                // rayleigh length in propagation direction
                static constexpr float_X rayleighLength_AXIS_1
                    = pmacc::math::Pi<float_X>::value * Waist::W0_AXIS_1 * Waist::W0_AXIS_1 / Base::WAVE_LENGTH;
                static constexpr float_X rayleighLength_AXIS_2
                    = pmacc::math::Pi<float_X>::value * Waist::W0_AXIS_2 * Waist::W0_AXIS_2 / Base::WAVE_LENGTH;
            };

            /** GaussianPulse incident E functor
             *
             * @tparam T_Params parameters
             * @tparam T_LongitudinalEnvelope class providing a static method getEnvelope(time)
             *  that defines laser temporal envelope.
             */
            template<typename T_Params, typename T_LongitudinalEnvelope>
            struct GaussianPulseFunctorIncidentE
                : public GaussianPulseUnitless<T_Params>
                , public incidentField::detail::BaseFunctorE<T_Params>
            {
                //! Unitless parameters type
                using Unitless = GaussianPulseUnitless<T_Params>;
                using LongitudinalEnvelope = T_LongitudinalEnvelope;

                //! Base functor type
                using Base = incidentField::detail::BaseFunctorE<T_Params>;

                /** Create a functor on the host side for the given time step
                 *
                 * @param currentStep current time step index, note that it is fractional
                 * @param unitField conversion factor from SI to internal units,
                 *                  fieldE_internal = fieldE_SI / unitField
                 */
                HINLINE GaussianPulseFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
                    : Base(currentStep, unitField)
                    , laguerre_norm(getLaguerreNorm())
                {
                    // This check is done here on HOST, since std::numeric_limits<float_X>::epsilon() does not
                    // compile on laserTransversal(), which is on DEVICE.
                    PMACC_VERIFY_MSG(
                        math::abs(laguerre_norm) > std::numeric_limits<float_X>::epsilon(),
                        "Sum of laguerreModes can not be 0.");
                }

                //!
                HINLINE static float_X getLaguerreNorm()
                {
                    float_X value = 0.0_X;
                    for(uint32_t m = 0; m < Unitless::laguerreModes.size(); ++m)
                        value += Unitless::laguerreModes[m];

                    return value;
                }

                /** Calculate incident field E value for the given position
                 *
                 * The transverse spatial laser modes are given as a decomposition of Gauss-Laguerre modes
                 * GLM(m,r,z) : Snorm * Sum_{m=0}^{m_max}{a_m * GLM(m,r,z)}
                 * with a_m being complex-valued coefficients: a_m := |a_m| * exp(I Arg(a_m) )
                 * |a_m| are equivalent to the laguerreModes vector entries.
                 * Arg(a_m) are equivalent to the laguerrePhases vector entries.
                 * The implicit pulse properties w0, lambda0, etc... equally apply to all GLM-modes.
                 * The on-axis, in-focus field value of the mode decomposition is normalized to unity:
                 * Snorm := 1 / ( Sum_{m=0}^{m_max}{GLM(m,0,0)} )
                 *
                 * Spatial mode: a_m * GLM(m,r,z) := w0/w(zeta) * L_m( 2*r^2/(w(zeta))^2 ) \
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
                 * R. Pausch et al. (2022), Modeling Beyond Gaussian Laser Pulses in Particle-in-Cell Simulations - The
                 * Impact of Higher Order Laser Modes, IEEE Advanced Accelerator Concepts Workshop (AAC), Long Island,
                 * NY, USA, pp. 1-5, doi: 10.1109/AAC55212.2022.10822876.
                 *
                 * F. Pampaloni et al. (2004), Gaussian, Hermite-Gaussian, and Laguerre-GaussianPulses: A
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
                    // get simulation time step
                    auto const time = this->currentTimeOrigin;

                    /* Calculate position of laser origin on the incidentField plane relative to the focus.
                     * Coordinate system is PIConGPUs and not the laser internal coordinate system where X is
                     * the propagation direction.
                     */
                    float3_X const originRelativeToFocus = this->origin - this->focus;

                    /* Relative distance of the origin from the laser focus, transformed into the laser
                     * coordination system where X is the propagation direction. */
                    float_X const distanceOriginRelativeToFocus
                        = pmacc::math::dot(originRelativeToFocus, this->getAxis0());

                    /* Shifting pulse for TIME_SHIFT, TIME_DELAY, and travel time from the focus to
                     * the origin in order to start with the front of the laser pulse at the origin. */
                    constexpr auto mue = LongitudinalEnvelope::TIME_SHIFT + Unitless::TIME_DELAY;
                    float_X const timeDelay = mue - (distanceOriginRelativeToFocus / sim.pic.getSpeedOfLight());

                    // transform current point of evaluation to laser internal coordinates
                    // The laser internal coordinate system's origin
                    // is in the focus and the laser propagates along the x-axis.
                    float3_X internalPosition = this->getInternalCoordinates(totalCellIdx);

                    // distance from focus in units of the Rayleigh length
                    float_X const normalizedDistanceFromFocus_AXIS_1
                        = internalPosition[0] / Unitless::rayleighLength_AXIS_1;

                    // beam waist at the generation plane so that at focus we will get W0
                    float_X const waist_AXIS_1
                        = Unitless::W0_AXIS_1
                          * math::sqrt(1._X + normalizedDistanceFromFocus_AXIS_1 * normalizedDistanceFromFocus_AXIS_1);

                    // inverse radius of curvature
                    float_X const inverseR_AXIS_1
                        = internalPosition[0]
                          / (internalPosition[0] * internalPosition[0]
                             + Unitless::rayleighLength_AXIS_1 * Unitless::rayleighLength_AXIS_1);

                    // Gouy phase
                    float_X gouy = .5_X * math::atan(normalizedDistanceFromFocus_AXIS_1);
                    float_X evaluationTime = time - timeDelay - internalPosition[0] / sim.pic.getSpeedOfLight()
                                             - internalPosition[1] * internalPosition[1] * inverseR_AXIS_1
                                                   / (2._X * sim.pic.getSpeedOfLight());

                    auto amplitudeExponent
                        = -internalPosition[1] * internalPosition[1] / (waist_AXIS_1 * waist_AXIS_1);

                    /* Compute amplitude and its reduction outside the focus due to defocussing. */
                    auto amplitudeAbs
                        = Unitless::AMPLITUDE
                          / math::sqrt(
                              math::sqrt(
                                  1._X + normalizedDistanceFromFocus_AXIS_1 * normalizedDistanceFromFocus_AXIS_1));

                    // add terms for 3D Pulse
                    if constexpr(simDim == DIM3)
                    {
                        float_X const normalizedDistanceFromFocus_AXIS_2
                            = internalPosition[0] / Unitless::rayleighLength_AXIS_2;
                        float_X const waist_AXIS_2
                            = Unitless::W0_AXIS_2
                              * math::sqrt(
                                  1._X + normalizedDistanceFromFocus_AXIS_2 * normalizedDistanceFromFocus_AXIS_2);
                        float_X const inverseR_AXIS_2
                            = internalPosition[0]
                              / (internalPosition[0] * internalPosition[0]
                                 + Unitless::rayleighLength_AXIS_2 * Unitless::rayleighLength_AXIS_2);

                        gouy += .5_X * math::atan(normalizedDistanceFromFocus_AXIS_2);
                        evaluationTime -= internalPosition[2] * internalPosition[2] * inverseR_AXIS_2
                                          / (2._X * sim.pic.getSpeedOfLight());
                        amplitudeExponent -= internalPosition[2] * internalPosition[2] / (waist_AXIS_2 * waist_AXIS_2);
                        amplitudeAbs /= math::sqrt(
                            math::sqrt(
                                1._X + normalizedDistanceFromFocus_AXIS_2 * normalizedDistanceFromFocus_AXIS_2));
                    }

                    auto const phase = Unitless::OMEGA0 * evaluationTime + gouy + Unitless::LASER_PHASE + phaseShift;
                    auto const eTemporal = LongitudinalEnvelope::getEnvelope(evaluationTime);

                    // initial values (m=0) for higher-order mode contributions
                    constexpr auto laguerreModes = Unitless::laguerreModes;
                    constexpr auto laguerrePhases = Unitless::laguerrePhases;
                    auto laguerre = laguerreModes[0] * math::cos(phase + laguerrePhases[0]);

                    if constexpr(laguerreModes.dim > uint32_t(1) && simDim == DIM3)
                    {
                        /** Apply higher-order Laguerre-Gaussian modes
                         * Since Laguerre-Gaussian modes are radially symmetric,
                         * it needs to be guaranteed that the laser waist sizes along AXIS_1 and AXIS_2 are equal.
                         * For this, assume W0_AXIS_1 and W0_AXIS_2 are positive, as they should.
                         * In case they are not, other things will break, such as computing a square root of a negative
                         * number. */
                        static_assert(
                            Unitless::W0_AXIS_1 - Unitless::W0_AXIS_2 < std::numeric_limits<float_X>::epsilon(),
                            "Waist sizes of the GaussianPulse must be equal along the transverse axes when using "
                            "Laguerre modes!");

                        // compute distance from beam axis
                        auto planeNoNormal = float3_X::create(1.0_X);
                        planeNoNormal[0] = 0.0_X;
                        auto const transversalDistanceSquared = pmacc::math::l2norm2(internalPosition * planeNoNormal);
                        auto const r2OverW2 = transversalDistanceSquared / (waist_AXIS_1 * waist_AXIS_1);

                        for(uint32_t m = 1; m < laguerreModes.size(); ++m)
                        {
                            // Add higher-order Laguerre-Gaussian mode contributions to transverse envelope

                            laguerre += laguerreModes[m] * simpleLaguerre(m, 2.0_X * r2OverW2)
                                        * math::cos(phase + 2._X * float_X(m) * gouy + laguerrePhases[m]);
                        }
                        laguerre /= laguerre_norm;
                    }

                    auto eSpatial = amplitudeAbs * math::exp(amplitudeExponent) * laguerre;
                    return eSpatial * eTemporal;
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

                /** Sum of amplitudes of Laguerre-Gaussian higher-modes
                 *
                 * Required for normalization.
                 */
                float_X const laguerre_norm;
            };
        } // namespace detail

        /** Gaussian temporal laser envelope
         *
         * @tparam T_Param param class
         */
        template<typename T_Param>
        struct GaussianPulseEnvelope : public detail::BaseParamUnitless<T_Param>
        {
            using Base = typename detail::BaseParamUnitless<T_Param>;
            using Unitless = detail::BaseParamUnitless<T_Param>;

            static constexpr float_X TIME_SHIFT = 0.5_X * Base::PULSE_INIT * Base::PULSE_DURATION;

            HDINLINE static float_X getEnvelope(float_X const time)
            {
                auto const exponent = time / (2.0_X * Unitless::PULSE_DURATION);

                return math::exp(-exponent * exponent);
            }

            HINLINE static std::string getName()
            {
                return "GaussianPulse";
            }
        };

        template<typename T_Func>
        struct FreeEnvelope
        {
            static constexpr float_X TIME_SHIFT = 0.0_X;

            HDINLINE static float_X getEnvelope(float_X const time)
            {
                return T_Func{}(time * sim.unit.time());
            }

            HINLINE static std::string getName()
            {
                return "FreeEnvelope";
            }
        };

        template<typename T_Params, typename T_LongitudinalEnvelope>
        struct GaussianPulse
        {
            using LongitudinalEnvelope = T_LongitudinalEnvelope;

            //! Get text name of the incident field profile
            HINLINE static std::string getName()
            {
                std::string name = "GaussianPulse_with_" + LongitudinalEnvelope::getName();
                return name;
            }
        };

    } // namespace profiles

    namespace traits::detail
    {
        /** Get type of incident field E functor for the GaussianPulse profile type
         *
         * @tparam T_Params parameters
         */
        template<typename T_Params, typename T_LongitudinalEnvelope>
        struct GetFunctorIncidentE<profiles::GaussianPulse<T_Params, T_LongitudinalEnvelope>>
        {
            using type = profiles::detail::GaussianPulseFunctorIncidentE<T_Params, T_LongitudinalEnvelope>;
        };

        /** Get type of incident field B functor for the GaussianPulse profile type
         *
         * Rely on SVEA to calculate value of B from E.
         *
         * @tparam T_Params parameters
         */
        template<typename T_Params, typename T_LongitudinalEnvelope>
        struct GetFunctorIncidentB<profiles::GaussianPulse<T_Params, T_LongitudinalEnvelope>>
        {
            using type = incidentField::detail::ApproximateIncidentB<
                typename GetFunctorIncidentE<profiles::GaussianPulse<T_Params, T_LongitudinalEnvelope>>::type>;
        };
    } // namespace traits::detail
} // namespace picongpu::fields::incidentField
