/* Copyright 2014-2024 Alexander Debus, Axel Huebl, Sergei Bastrakov
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
#include "picongpu/fields/YeeCell.hpp"
#include "picongpu/fields/background/templates/twtstight/EField.hpp"
#include "picongpu/fields/background/templates/twtstight/GetInitialTimeDelay_SI.tpp"
#include "picongpu/fields/background/templates/twtstight/getFieldPositions_SI.tpp"
#include "picongpu/fields/background/templates/twtstight/twtstight.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/math/Complex.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/types.hpp>

namespace picongpu
{
    /* Load pre-defined background field */
    namespace templates
    {
        /* Traveling-wave Thomson scattering laser pulse */
        namespace twtstight
        {
            HINLINE
            EField::EField(
                float_64 const focus_y_SI,
                float_64 const wavelength_SI,
                float_64 const pulselength_SI,
                float_64 const w_x_SI,
                float_X const phi,
                float_X const beta_0,
                float_64 const tdelay_user_SI,
                bool const auto_tdelay,
                float_X const polAngle)
                : focus_y_SI(focus_y_SI)
                , wavelength_SI(wavelength_SI)
                , pulselength_SI(pulselength_SI)
                , w_x_SI(w_x_SI)
                , phi(phi)
                , phiPositive(float_X(1.0))
                , beta_0(beta_0)
                , tdelay_user_SI(tdelay_user_SI)
                , dt(sim.si.getDt())
                , unit_length(sim.unit.length())
                , auto_tdelay(auto_tdelay)
                , polAngle(polAngle)
            {
                /* Note: Enviroment-objects cannot be instantiated on CUDA GPU device. Since this is done
                         on host (see fieldBackground.param), this is no problem.
                 */
                SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                halfSimSize = subGrid.getGlobalDomain().size / 2;
                tdelay = detail::getInitialTimeDelay_SI(
                    auto_tdelay,
                    tdelay_user_SI,
                    halfSimSize,
                    pulselength_SI,
                    focus_y_SI,
                    phi,
                    beta_0);
                if(phi < 0.0_X)
                    phiPositive = float_X(-1.0);
            }

            template<>
            HDINLINE float3_X EField::getTWTSEfield_Normalized<DIM3>(
                pmacc::math::Vector<floatD_64, detail::numComponents> const& eFieldPositions_SI,
                float_64 const time) const
            {
                using PosVecVec = pmacc::math::Vector<float3_64, detail::numComponents>;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    for(uint32_t i = 0; i < simDim; ++i)
                        pos[k][i] = eFieldPositions_SI[k][i];
                }

                /* E-field normalized to the peak amplitude. */
                return float3_X(
                    float_X(calcTWTSEx(pos[0], time)),
                    float_X(calcTWTSEy(pos[1], time)),
                    float_X(calcTWTSEz(pos[2], time)));
            }

            template<>
            HDINLINE float3_X EField::getTWTSEfield_Normalized<DIM2>(
                pmacc::math::Vector<floatD_64, detail::numComponents> const& eFieldPositions_SI,
                float_64 const time) const
            {
                using PosVecVec = pmacc::math::Vector<float3_64, detail::numComponents>;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    /* 2D (y,z) vectors are mapped on 3D (x,y,z) vectors. */
                    for(uint32_t i = 0; i < DIM2; ++i)
                    {
                        pos[k][i + 1] = eFieldPositions_SI[k][i];
                    }
                }

                /* E-field normalized to the peak amplitude. */
                return float3_X(calcTWTSEx(pos[0], time), calcTWTSEy(pos[1], time), calcTWTSEz(pos[2], time));
            }

            HDINLINE float3_X EField::operator()(DataSpace<simDim> const& cellIdx, uint32_t const currentStep) const
            {
                traits::FieldPosition<fields::YeeCell, FieldE> const fieldPosE;
                return getValue(precisionCast<float_X>(cellIdx), fieldPosE(), static_cast<float_X>(currentStep));
            }

            HDINLINE
            float3_X EField::operator()(floatD_X const& cellIdx, float_X const currentStep) const
            {
                pmacc::math::Vector<floatD_X, detail::numComponents> zeroShifts;
                for(uint32_t component = 0; component < detail::numComponents; ++component)
                    zeroShifts[component] = floatD_X::create(0.0);
                return getValue(cellIdx, zeroShifts, currentStep);
            }

            HDINLINE
            float3_X EField::getValue(
                floatD_X const& cellIdx,
                pmacc::math::Vector<floatD_X, detail::numComponents> const& extraShifts,
                float_X const currentStep) const
            {
                float_64 const time_SI = float_64(currentStep) * dt - tdelay;

                pmacc::math::Vector<floatD_64, detail::numComponents> const eFieldPositions_SI
                    = detail::getFieldPositions_SI(cellIdx, halfSimSize, extraShifts, unit_length, focus_y_SI, phi);

                /* Single TWTS-Pulse */
                return getTWTSEfield_Normalized<simDim>(eFieldPositions_SI, time_SI);
            }

            template<uint32_t T_component>
            HDINLINE float_X EField::getComponent(floatD_X const& cellIdx, float_X const currentStep) const
            {
                // The optimized way is only implemented for 3D, fall back to full field calculation in 2d
                if constexpr(simDim == DIM3)
                {
                    float_64 const time_SI = float_64(currentStep) * dt - tdelay;
                    pmacc::math::Vector<floatD_X, detail::numComponents> zeroShifts;
                    for(uint32_t component = 0; component < detail::numComponents; ++component)
                        zeroShifts[component] = floatD_X::create(0.0);
                    pmacc::math::Vector<floatD_64, detail::numComponents> const eFieldPositions_SI
                        = detail::getFieldPositions_SI(cellIdx, halfSimSize, zeroShifts, unit_length, focus_y_SI, phi);
                    // Explicitly use a 3D vector so that this function compiles for 2D as well
                    auto const pos = float3_64{
                        eFieldPositions_SI[T_component][0],
                        eFieldPositions_SI[T_component][1],
                        eFieldPositions_SI[T_component][2]};
                    float_X sinPhi;
                    float_X cosPhi;
                    pmacc::math::sincos(phi, sinPhi, cosPhi);

                    if constexpr(T_component == 0)
                        return static_cast<float_X>(calcTWTSEx(pos, time_SI));
                    else
                    {
                        if constexpr(T_component == 1)
                        {
                            return static_cast<float_X>(calcTWTSEy(pos, time_SI));
                        }
                        if constexpr(T_component == 2)
                        {
                            return static_cast<float_X>(calcTWTSEz(pos, time_SI));
                        }
                    }

                    // we should never be here
                    return NAN;
                }
                if constexpr(simDim != DIM3)
                    return (*this)(cellIdx, currentStep)[T_component];
            }

            /** Calculate the Ex(r,t) field here
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations) for calculating
             *             the field */
            HDINLINE EField::float_T EField::calcTWTSEx(float3_64 const& pos, float_64 const time) const
            {
                using complex_T = alpaka::Complex<float_T>;
                using complex_64 = alpaka::Complex<float_64>;

                /* Propagation speed of overlap normalized to the speed of light [Default: beta0=1.0] */
                auto const beta0 = float_T(beta_0);
                /* If phi < 0 the formulas below are not directly applicable.
                 * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
                 * z-axis of the coordinate system in this function.
                 */
                auto const phiReal = float_T(math::abs(phi));
                float_T sinPhiReal;
                float_T cosPhiReal;
                pmacc::math::sincos(phiReal, sinPhiReal, cosPhiReal);
                float_T const alphaTilt = math::atan2(float_T(1.0) - beta0 * cosPhiReal, beta0 * sinPhiReal);
                /* Definition of the laser pulse front tilt angle for the laser field below.
                 *
                 * For beta0 = 1.0, this is equivalent to our standard definition. Question: Why is the
                 * local "phi_T" not equal in value to the object member "phiReal" or "phi"?
                 * Because the standard TWTS pulse is defined for beta0 = 1.0 and in the coordinate-system
                 * of the TWTS model phi is responsible for pulse front tilt and dispersion only. Hence
                 * the dispersion will (although physically correct) be slightly off the ideal TWTS
                 * pulse for beta0 != 1.0. This only shows that this TWTS pulse is primarily designed for
                 * scenarios close to beta0 = 1.
                 */
                float_T const phiT = float_T(2.0) * alphaTilt;

                /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for
                 * documentation purposes.
                 * float_T const eta = (PI / 2) - (phiReal - alphaTilt);
                 */

                auto const cspeed = float_T(sim.si.getSpeedOfLight() / sim.unit.speed());
                auto const lambda0 = float_T(wavelength_SI / sim.unit.length());
                float_T const omega0 = float_T(2.0 * PI) * cspeed / lambda0;
                /* factor 2  in tauG arises from definition convention in laser formula */
                auto const tauG = float_T(pulselength_SI * 2.0 / sim.unit.time());
                /* w0 is wx here --> w0 could be replaced by wx */
                auto const w0 = float_T(w_x_SI / sim.unit.length());
                auto const k = float_T(2.0 * PI / lambda0);

                /* In order to calculate in single-precision and in order to account for errors in
                 * the approximations far from the coordinate origin, we use the wavelength-periodicity and
                 * the known propagation direction for realizing the laser pulse using relative coordinates
                 * (i.e. from a finite coordinate range) only. All these quantities have to be calculated
                 * in double precision.
                 */
                float_64 sinPhiVal;
                float_64 cosPhiVal;
                pmacc::math::sincos(precisionCast<float_64>(phi), sinPhiVal, cosPhiVal);
                float_64 const tanAlpha = (1.0 - beta_0 * cosPhiVal) / (beta_0 * sinPhiVal);
                float_64 const tanFocalLine = math::tan(PI / 2.0 - phi);
                float_64 const deltaT = wavelength_SI / sim.si.getSpeedOfLight() * (1.0 + tanAlpha / tanFocalLine);
                float_64 const deltaY = wavelength_SI * cosPhiVal + wavelength_SI * sinPhiVal * sinPhiVal / cosPhiVal;
                float_64 const numberOfPeriods = math::floor(time / deltaT);
                auto const timeMod = float_T(time - numberOfPeriods * deltaT);
                auto const yMod = float_T(pos.y() - numberOfPeriods * deltaY);

                auto const x = float_T(phiPositive * pos.x() / sim.unit.length());
                auto const y = float_T(yMod / sim.unit.length());
                auto const z = float_T(phiPositive * pos.z() / sim.unit.length());
                auto const t = float_T(timeMod / sim.unit.time());
                /* To avoid underflows in computation, fields are set to zero
                 * before and after the respective TWTS pulse envelope.
                 */
                if(math::abs(y - z * math::tan(phiT / float_T(2.0)) - (cspeed * t)) > (numSigmas * tauG * cspeed))
                    return float_T(0.0);

                /* Calculating shortcuts for speeding up field calculation */
                float_T sinPhi;
                float_T cosPhi;
                pmacc::math::sincos(phiT, sinPhi, cosPhi);
                float_T const cosPhi2 = math::cos(phiT / float_T(2.0));
                float_T const tanPhi2 = math::tan(phiT / float_T(2.0));
                float_T const sinPhi_2 = sinPhi * sinPhi;
                float_T const cosPhi_2 = cosPhi * cosPhi;
                float_T const sinPolAngle = math::sin(polAngle);
                float_T const cosPolAngle = math::cos(polAngle);

                complex_T I = complex_T(0, 1);
                float_T const x2 = x * x;
                float_T const tauG2 = tauG * tauG;
                float_T const psi0 = float_T(2.0) / k;
                float_T const w02 = w0 * w0;

                complex_T const nu = (y * cosPhi - z * sinPhi) / cspeed;
                complex_T const xi = (-z * cosPhi - y * sinPhi) * tanPhi2 / cspeed;
                complex_T const rhom
                    = math::sqrt(x2 + pmacc::math::cPow(-z - complex_T(0, 0.5) * k * w02, static_cast<uint32_t>(2u)));
                complex_T const rhom2 = rhom * rhom;
                complex_T const Xm = -z - complex_T(0, 0.5) * k * w02;
                float_T const besselI0const = pmacc::math::bessel::i0(k * k * sinPhi * w02 / float_T(2.0));
                complex_T const besselJ0const = pmacc::math::bessel::j0(k * rhom * sinPhi);
                complex_T const besselJ1const = pmacc::math::bessel::j1(k * rhom * sinPhi);

                complex_T const zeroOrder = tauG * math::sqrt(omega0 * (float_T(1.0) + cosPhi))
                    / (math::sqrt(float_T(2.0))
                       * math::exp(
                           omega0 * pmacc::math::cPow((t - nu + xi) * cosPhi2, static_cast<uint32_t>(2u))
                           / (+(omega0 * tauG2) / float_T(2.0) - complex_T(0, 1) * nu
                              + ((omega0 * tauG2) / float_T(2.0) + complex_T(0, 1) * (nu - xi)) * cosPhi))
                       * math::sqrt(
                           +(omega0 * tauG2) / float_T(2.0) - I * nu
                           + ((omega0 * tauG2) / float_T(2.0) + I * (nu - xi)) * cosPhi));

                complex_T const result
                    = (complex_T(0, 0.25) * math::exp(I * (omega0 * t - k * y * cosPhi)) * zeroOrder
                       * (k * rhom * besselJ0const
                              * ((rhom2 - x2 + x * Xm * cosPhi) * sinPolAngle * sinPhi_2
                                 + cosPolAngle
                                     * (rhom2 + rhom2 * cosPhi_2 - x2 * sinPhi_2 - x * Xm * cosPhi * sinPhi_2))
                          + besselJ1const * sinPhi
                              * (+sinPolAngle
                                     * (-rhom2 + float_T(2.0) * x2 - I * k * rhom2 * Xm * sinPhi
                                        + x * cosPhi * (float_T(-2.0) * Xm - I * k * rhom2 * sinPhi))
                                 + cosPolAngle
                                     * (-rhom2 + float_T(2.0) * x2 + I * k * rhom2 * Xm * sinPhi
                                        + x * cosPhi * (float_T(+2.0) * Xm + I * k * rhom2 * sinPhi))))
                       * psi0)
                    / (rhom * rhom2 * besselI0const);

                /* A 180deg-rotation of the field vector around the y-axis
                 * leads to a sign flip in the x- and z- components, respectively.
                 * This is implemented by multiplying the result by "phiPositive".
                 */
                return phiPositive * result.real();
            }

            /** Calculate the Ey(r,t) field here
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations) for calculating
             *             the field */
            HDINLINE EField::float_T EField::calcTWTSEy(float3_64 const& pos, float_64 const time) const
            {
                using complex_T = alpaka::Complex<float_T>;
                using complex_64 = alpaka::Complex<float_64>;

                /* Propagation speed of overlap normalized to the speed of light [Default: beta0=1.0] */
                auto const beta0 = float_T(beta_0);
                /* If phi < 0 the formulas below are not directly applicable.
                 * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
                 * z-axis of the coordinate system in this function.
                 */
                auto const phiReal = float_T(math::abs(phi));
                float_T sinPhiReal;
                float_T cosPhiReal;
                pmacc::math::sincos(phiReal, sinPhiReal, cosPhiReal);
                float_T const alphaTilt = math::atan2(float_T(1.0) - beta0 * cosPhiReal, beta0 * sinPhiReal);
                /* Definition of the laser pulse front tilt angle for the laser field below.
                 *
                 * For beta0 = 1.0, this is equivalent to our standard definition. Question: Why is the
                 * local "phi_T" not equal in value to the object member "phiReal" or "phi"?
                 * Because the standard TWTS pulse is defined for beta0 = 1.0 and in the coordinate-system
                 * of the TWTS model phi is responsible for pulse front tilt and dispersion only. Hence
                 * the dispersion will (although physically correct) be slightly off the ideal TWTS
                 * pulse for beta0 != 1.0. This only shows that this TWTS pulse is primarily designed for
                 * scenarios close to beta0 = 1.
                 */
                float_T const phiT = float_T(2.0) * alphaTilt;

                /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for
                 * documentation purposes.
                 * float_T const eta = (PI / 2) - (phiReal - alphaTilt);
                 */

                auto const cspeed = float_T(sim.si.getSpeedOfLight() / sim.unit.speed());
                auto const lambda0 = float_T(wavelength_SI / sim.unit.length());
                float_T const omega0 = float_T(2.0 * PI) * cspeed / lambda0;
                /* factor 2  in tauG arises from definition convention in laser formula */
                auto const tauG = float_T(pulselength_SI * 2.0 / sim.unit.time());
                /* w0 is wx here --> w0 could be replaced by wx */
                auto const w0 = float_T(w_x_SI / sim.unit.length());
                auto const k = float_T(2.0 * PI / lambda0);

                /* In order to calculate in single-precision and in order to account for errors in
                 * the approximations far from the coordinate origin, we use the wavelength-periodicity and
                 * the known propagation direction for realizing the laser pulse using relative coordinates
                 * (i.e. from a finite coordinate range) only. All these quantities have to be calculated
                 * in double precision.
                 */
                float_64 sinPhiVal;
                float_64 cosPhiVal;
                pmacc::math::sincos(precisionCast<float_64>(phi), sinPhiVal, cosPhiVal);
                float_64 const tanAlpha = (1.0 - beta_0 * cosPhiVal) / (beta_0 * sinPhiVal);
                float_64 const tanFocalLine = math::tan(PI / 2.0 - phi);
                float_64 const deltaT = wavelength_SI / sim.si.getSpeedOfLight() * (1.0 + tanAlpha / tanFocalLine);
                float_64 const deltaY = wavelength_SI * cosPhiVal + wavelength_SI * sinPhiVal * sinPhiVal / cosPhiVal;
                float_64 const numberOfPeriods = math::floor(time / deltaT);
                auto const timeMod = float_T(time - numberOfPeriods * deltaT);
                auto const yMod = float_T(pos.y() - numberOfPeriods * deltaY);

                auto const x = float_T(phiPositive * pos.x() / sim.unit.length());
                auto const y = float_T(yMod / sim.unit.length());
                auto const z = float_T(phiPositive * pos.z() / sim.unit.length());
                auto const t = float_T(timeMod / sim.unit.time());
                /* To avoid underflows in computation, fields are set to zero
                 * before and after the respective TWTS pulse envelope.
                 */
                if(math::abs(y - z * math::tan(phiT / float_T(2.0)) - (cspeed * t)) > (numSigmas * tauG * cspeed))
                    return float_T(0.0);

                /* Calculating shortcuts for speeding up field calculation */
                float_T sinPhi;
                float_T cosPhi;
                pmacc::math::sincos(phiT, sinPhi, cosPhi);
                float_T const cosPhi2 = math::cos(phiT / float_T(2.0));
                float_T const tanPhi2 = math::tan(phiT / float_T(2.0));
                float_T const sinPhi_2 = sinPhi * sinPhi;
                float_T const cosPhi_2 = cosPhi * cosPhi;
                float_T const sinPolAngle = math::sin(polAngle);
                float_T const cosPolAngle = math::cos(polAngle);

                complex_T I = complex_T(0, 1);
                float_T const x2 = x * x;
                float_T const tauG2 = tauG * tauG;
                float_T const psi0 = float_T(2.0) / k;
                float_T const w02 = w0 * w0;

                complex_T const nu = (y * cosPhi - z * sinPhi) / cspeed;
                complex_T const xi = (-z * cosPhi - y * sinPhi) * tanPhi2 / cspeed;
                complex_T const rhom
                    = math::sqrt(x2 + pmacc::math::cPow(-z - complex_T(0, 0.5) * k * w02, static_cast<uint32_t>(2u)));
                complex_T const Xm = -z - complex_T(0, 0.5) * k * w02;
                float_T const besselI0const = pmacc::math::bessel::i0(k * k * sinPhi * w02 / float_T(2.0));
                complex_T const besselJ0const = pmacc::math::bessel::j0(k * rhom * sinPhi);
                complex_T const besselJ1const = pmacc::math::bessel::j1(k * rhom * sinPhi);

                complex_T const zeroOrder = tauG * math::sqrt(omega0 * (float_T(1.0) + cosPhi))
                    / (math::sqrt(float_T(2.0))
                       * math::exp(
                           omega0 * pmacc::math::cPow((t - nu + xi) * cosPhi2, static_cast<uint32_t>(2u))
                           / (+(omega0 * tauG2) / float_T(2.0) - complex_T(0, 1) * nu
                              + ((omega0 * tauG2) / float_T(2.0) + complex_T(0, 1) * (nu - xi)) * cosPhi))
                       * math::sqrt(
                           +(omega0 * tauG2) / float_T(2.0) - I * nu
                           + ((omega0 * tauG2) / float_T(2.0) + I * (nu - xi)) * cosPhi));

                complex_T const result = (math::exp(I * (omega0 * t - k * y * cosPhi)) * k * zeroOrder * sinPhi
                                          * (besselJ1const
                                                 * (cosPolAngle * (Xm - float_T(2.0) * x * cosPhi - Xm * cosPhi_2)
                                                    + Xm * (float_T(1.0) + cosPhi_2) * sinPolAngle)
                                             + I * rhom * besselJ0const * (cosPolAngle - sinPolAngle) * sinPhi_2)
                                          * psi0)
                    / (float_T(4.0) * rhom * besselI0const);

                return result.real();
            }

            /** Calculate the Ez(r,t) field here
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations) for calculating
             *             the field */
            HDINLINE EField::float_T EField::calcTWTSEz(float3_64 const& pos, float_64 const time) const
            {
                using complex_T = alpaka::Complex<float_T>;
                using complex_64 = alpaka::Complex<float_64>;

                /* Propagation speed of overlap normalized to the speed of light [Default: beta0=1.0] */
                auto const beta0 = float_T(beta_0);
                /* If phi < 0 the formulas below are not directly applicable.
                 * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
                 * z-axis of the coordinate system in this function.
                 */
                auto const phiReal = float_T(math::abs(phi));
                float_T sinPhiReal;
                float_T cosPhiReal;
                pmacc::math::sincos(phiReal, sinPhiReal, cosPhiReal);
                float_T const alphaTilt = math::atan2(float_T(1.0) - beta0 * cosPhiReal, beta0 * sinPhiReal);
                /* Definition of the laser pulse front tilt angle for the laser field below.
                 *
                 * For beta0 = 1.0, this is equivalent to our standard definition. Question: Why is the
                 * local "phi_T" not equal in value to the object member "phiReal" or "phi"?
                 * Because the standard TWTS pulse is defined for beta0 = 1.0 and in the coordinate-system
                 * of the TWTS model phi is responsible for pulse front tilt and dispersion only. Hence
                 * the dispersion will (although physically correct) be slightly off the ideal TWTS
                 * pulse for beta0 != 1.0. This only shows that this TWTS pulse is primarily designed for
                 * scenarios close to beta0 = 1.
                 */
                float_T const phiT = float_T(2.0) * alphaTilt;

                /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for
                 * documentation purposes.
                 * float_T const eta = (PI / 2) - (phiReal - alphaTilt);
                 */

                auto const cspeed = float_T(sim.si.getSpeedOfLight() / sim.unit.speed());
                auto const lambda0 = float_T(wavelength_SI / sim.unit.length());
                float_T const omega0 = float_T(2.0 * PI) * cspeed / lambda0;
                /* factor 2  in tauG arises from definition convention in laser formula */
                auto const tauG = float_T(pulselength_SI * 2.0 / sim.unit.time());
                /* w0 is wx here --> w0 could be replaced by wx */
                auto const w0 = float_T(w_x_SI / sim.unit.length());
                auto const k = float_T(2.0 * PI / lambda0);

                /* In order to calculate in single-precision and in order to account for errors in
                 * the approximations far from the coordinate origin, we use the wavelength-periodicity and
                 * the known propagation direction for realizing the laser pulse using relative coordinates
                 * (i.e. from a finite coordinate range) only. All these quantities have to be calculated
                 * in double precision.
                 */
                float_64 sinPhiVal;
                float_64 cosPhiVal;
                pmacc::math::sincos(precisionCast<float_64>(phi), sinPhiVal, cosPhiVal);
                float_64 const tanAlpha = (1.0 - beta_0 * cosPhiVal) / (beta_0 * sinPhiVal);
                float_64 const tanFocalLine = math::tan(PI / 2.0 - phi);
                float_64 const deltaT = wavelength_SI / sim.si.getSpeedOfLight() * (1.0 + tanAlpha / tanFocalLine);
                float_64 const deltaY = wavelength_SI * cosPhiVal + wavelength_SI * sinPhiVal * sinPhiVal / cosPhiVal;
                float_64 const numberOfPeriods = math::floor(time / deltaT);
                auto const timeMod = float_T(time - numberOfPeriods * deltaT);
                auto const yMod = float_T(pos.y() - numberOfPeriods * deltaY);

                auto const x = float_T(phiPositive * pos.x() / sim.unit.length());
                auto const y = float_T(yMod / sim.unit.length());
                auto const z = float_T(phiPositive * pos.z() / sim.unit.length());
                auto const t = float_T(timeMod / sim.unit.time());
                /* To avoid underflows in computation, fields are set to zero
                 * before and after the respective TWTS pulse envelope.
                 */
                if(math::abs(y - z * math::tan(phiT / float_T(2.0)) - (cspeed * t)) > (numSigmas * tauG * cspeed))
                    return float_T(0.0);

                /* Calculating shortcuts for speeding up field calculation */
                float_T sinPhi;
                float_T cosPhi;
                pmacc::math::sincos(phiT, sinPhi, cosPhi);
                float_T const cosPhi2 = math::cos(phiT / float_T(2.0));
                float_T const tanPhi2 = math::tan(phiT / float_T(2.0));
                float_T const sinPhi_2 = sinPhi * sinPhi;
                float_T const sinPolAngle = math::sin(polAngle);
                float_T const cosPolAngle = math::cos(polAngle);
                float_T const sin2Phi = math::sin(float_T(2.0) * phiT);

                complex_T I = complex_T(0, 1);
                float_T const x2 = x * x;
                float_T const tauG2 = tauG * tauG;
                float_T const psi0 = float_T(2.0) / k;
                float_T const w02 = w0 * w0;

                complex_T const nu = (y * cosPhi - z * sinPhi) / cspeed;
                complex_T const xi = (-z * cosPhi - y * sinPhi) * tanPhi2 / cspeed;
                complex_T const rhom
                    = math::sqrt(x2 + pmacc::math::cPow(-z - complex_T(0, 0.5) * k * w02, static_cast<uint32_t>(2u)));
                complex_T const rhom2 = rhom * rhom;
                complex_T const Xm = -z - complex_T(0, 0.5) * k * w02;
                complex_T const Xm2 = Xm * Xm;
                float_T const besselI0const = pmacc::math::bessel::i0(k * k * sinPhi * w02 / float_T(2.0));
                complex_T const besselJ0const = pmacc::math::bessel::j0(k * rhom * sinPhi);
                complex_T const besselJ1const = pmacc::math::bessel::j1(k * rhom * sinPhi);

                complex_T const zeroOrder = tauG * math::sqrt(omega0 * (float_T(1.0) + cosPhi))
                    / (math::sqrt(float_T(2.0))
                       * math::exp(
                           omega0 * pmacc::math::cPow((t - nu + xi) * cosPhi2, static_cast<uint32_t>(2u))
                           / (+(omega0 * tauG2) / float_T(2.0) - complex_T(0, 1) * nu
                              + ((omega0 * tauG2) / float_T(2.0) + complex_T(0, 1) * (nu - xi)) * cosPhi))
                       * math::sqrt(
                           +(omega0 * tauG2) / float_T(2.0) - I * nu
                           + ((omega0 * tauG2) / float_T(2.0) + I * (nu - xi)) * cosPhi));

                complex_T const result
                    = (complex_T(0, 0.125) * math::exp(I * (omega0 * t - k * y * cosPhi)) * zeroOrder
                       * (float_T(2.0) * k * rhom * besselJ0const
                              * (x * Xm * (cosPolAngle + sinPolAngle) * sinPhi_2
                                 + cosPhi
                                     * (Xm2 * cosPolAngle * sinPhi_2
                                        + sinPolAngle * (float_T(2.0) * rhom2 - Xm2 * sinPhi_2)))
                          + besselJ1const * sinPhi
                              * (cosPolAngle
                                     * (float_T(-4.0) * x * Xm + float_T(2.0) * (rhom2 - float_T(2.0) * Xm2) * cosPhi
                                        + complex_T(0, 2) * k * rhom2 * (x - Xm * cosPhi) * sinPhi)
                                 + sinPolAngle
                                     * (float_T(-4.0) * x * Xm - float_T(2.0) * (rhom2 - float_T(2.0) * Xm2) * cosPhi
                                        - complex_T(0, 2) * k * rhom2 * x * sinPhi + I * k * rhom2 * Xm * sin2Phi)))
                       * psi0)
                    / (rhom * rhom2 * besselI0const);

                /* A 180deg-rotation of the field vector around the y-axis
                 * leads to a sign flip in the x- and z- components, respectively.
                 * This is implemented by multiplying the result by "phiPositive".
                 */
                return phiPositive * result.real();
            }

        } /* namespace twtstight */
    } /* namespace templates */
} /* namespace picongpu */
