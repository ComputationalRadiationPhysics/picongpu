/* Copyright 2014-2021 Alexander Debus, Axel Huebl
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

#include <pmacc/types.hpp>
#include "picongpu/simulation_defines.hpp"

#include <pmacc/math/Vector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/math/Complex.hpp>

#include "picongpu/fields/background/templates/twtsfast/RotateField.tpp"
#include "picongpu/fields/background/templates/twtsfast/GetInitialTimeDelay_SI.tpp"
#include "picongpu/fields/background/templates/twtsfast/getFieldPositions_SI.tpp"
#include "picongpu/fields/background/templates/twtsfast/EField.hpp"
#include "picongpu/fields/CellType.hpp"

namespace picongpu
{
    /* Load pre-defined background field */
    namespace templates
    {
        /* Traveling-wave Thomson scattering laser pulse */
        namespace twtsfast
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
                PolarizationType const pol)
                : focus_y_SI(focus_y_SI)
                , wavelength_SI(wavelength_SI)
                , pulselength_SI(pulselength_SI)
                , w_x_SI(w_x_SI)
                , phi(phi)
                , beta_0(beta_0)
                , tdelay_user_SI(tdelay_user_SI)
                , dt(SI::DELTA_T_SI)
                , unit_length(UNIT_LENGTH)
                , auto_tdelay(auto_tdelay)
                , pol(pol)
                , phiPositive(float_X(1.0))
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
                float3_64 pos(float3_64::create(0.0));
                for(uint32_t i = 0; i < simDim; ++i)
                    pos[i] = eFieldPositions_SI[0][i];
                return float3_X(float_X(calcTWTSEx(pos, time)), 0.0_X, 0.0_X);
            }

            template<>
            HDINLINE float3_X EField::getTWTSEfield_Normalized_Ey<DIM3>(
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

                /* Calculate Ey-component with the intra-cell offset of a Ey-field */
                float_64 const Ey_Ey = calcTWTSEy(pos[1], time);
                /* Calculate Ey-component with the intra-cell offset of a Ez-field */
                float_64 const Ey_Ez = calcTWTSEy(pos[2], time);

                /* Since we rotated all position vectors before calling calcTWTSEy,
                 * we need to back-rotate the resulting E-field vector.
                 *
                 * RotationMatrix[-(PI/2+phi)].(Ey,Ez) for rotating back the field-vectors.
                 */
                float_X sinPhi;
                float_X cosPhi;
                pmacc::math::sincos(phi, sinPhi, cosPhi);
                float_X const Ey_rot = -sinPhi * float_X(Ey_Ey);
                float_X const Ez_rot = -cosPhi * float_X(Ey_Ez);

                /* Finally, the E-field normalized to the peak amplitude. */
                return float3_X(0.0_X, Ey_rot, Ez_rot);
            }

            template<>
            HDINLINE float3_X EField::getTWTSEfield_Normalized<DIM2>(
                pmacc::math::Vector<floatD_64, detail::numComponents> const& eFieldPositions_SI,
                float_64 const time) const
            {
                /* Ex->Ez, so also the grid cell offset for Ez has to be used. */
                float3_64 pos(float3_64::create(0.0));
                /* 2D (y,z) vectors are mapped on 3D (x,y,z) vectors. */
                for(uint32_t i = 0; i < DIM2; ++i)
                    pos[i + 1] = eFieldPositions_SI[2][i];
                return float3_X(0.0_X, 0.0_X, float_X(calcTWTSEx(pos, time)));
            }

            template<>
            HDINLINE float3_X EField::getTWTSEfield_Normalized_Ey<DIM2>(
                pmacc::math::Vector<floatD_64, detail::numComponents> const& eFieldPositions_SI,
                float_64 const time) const
            {
                using PosVecVec = pmacc::math::Vector<float3_64, detail::numComponents>;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                /* The 2D output of getFieldPositions_SI only returns
                 * the y- and z-component of a 3D vector.
                 */
                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    for(uint32_t i = 0; i < DIM2; ++i)
                        pos[k][i + 1] = eFieldPositions_SI[k][i];
                }

                /* Ey->Ey, but grid cell offsets for Ex and Ey have to be used.
                 *
                 * Calculate Ey-component with the intra-cell offset of a Ey-field
                 */
                float_64 const Ey_Ey = calcTWTSEy(pos[1], time);
                /* Calculate Ey-component with the intra-cell offset of a Ex-field */
                float_64 const Ey_Ex = calcTWTSEy(pos[0], time);

                /* Since we rotated all position vectors before calling calcTWTSEy,
                 * we need to back-rotate the resulting E-field vector.
                 *
                 * RotationMatrix[-(PI / 2+phi)].(Ey,Ex) for rotating back the field-vectors.
                 */
                float_X sinPhi;
                float_X cosPhi;
                pmacc::math::sincos(phi, sinPhi, cosPhi);
                float_X const Ey_rot = -sinPhi * float_X(Ey_Ey);
                float_X const Ex_rot = -cosPhi * float_X(Ey_Ex);

                /* Finally, the E-field normalized to the peak amplitude. */
                return float3_X(Ex_rot, Ey_rot, 0.0_X);
            }

            HDINLINE float3_X EField::operator()(DataSpace<simDim> const& cellIdx, uint32_t const currentStep) const
            {
                float_64 const time_SI = float_64(currentStep) * dt - tdelay;
                traits::FieldPosition<fields::CellType, FieldE> const fieldPosE;

                pmacc::math::Vector<floatD_64, detail::numComponents> const eFieldPositions_SI
                    = detail::getFieldPositions_SI(cellIdx, halfSimSize, fieldPosE(), unit_length, focus_y_SI, phi);

                /* Single TWTS-Pulse */
                switch(pol)
                {
                case LINEAR_X:
                    return getTWTSEfield_Normalized<simDim>(eFieldPositions_SI, time_SI);

                case LINEAR_YZ:
                    return getTWTSEfield_Normalized_Ey<simDim>(eFieldPositions_SI, time_SI);
                }
                return getTWTSEfield_Normalized<simDim>(eFieldPositions_SI, time_SI); // defensive default
            }

            /** Calculate the Ex(r,t) field here
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations) for calculating
             *             the field */
            HDINLINE EField::float_T EField::calcTWTSEx(float3_64 const& pos, float_64 const time) const
            {
                using complex_T = pmacc::math::Complex<float_T>;
                using complex_64 = pmacc::math::Complex<float_64>;

                /* Propagation speed of overlap normalized to the speed of light [Default: beta0=1.0] */
                float_T const beta0 = float_T(beta_0);
                /* If phi < 0 the formulas below are not directly applicable.
                 * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
                 * z-axis of the coordinate system in this function.
                 */
                float_T const phiReal = float_T(math::abs(phi));
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

                float_T const cspeed = float_T(SI::SPEED_OF_LIGHT_SI / UNIT_SPEED);
                float_T const lambda0 = float_T(wavelength_SI / UNIT_LENGTH);
                float_T const om0 = float_T(2.0 * PI) * cspeed / lambda0;
                /* factor 2  in tauG arises from definition convention in laser formula */
                float_T const tauG = float_T(pulselength_SI * 2.0 / UNIT_TIME);
                /* w0 is wx here --> w0 could be replaced by wx */
                float_T const w0 = float_T(w_x_SI / UNIT_LENGTH);
                float_T const rho0 = float_T(PI * w0 * w0 / lambda0);
                float_T const k = float_T(2.0 * PI / lambda0);

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
                float_64 const deltaT = wavelength_SI / SI::SPEED_OF_LIGHT_SI * (1.0 + tanAlpha / tanFocalLine);
                float_64 const deltaY = wavelength_SI / tanFocalLine;
                float_64 const deltaZ = -wavelength_SI;
                float_64 const numberOfPeriods = math::floor(time / deltaT);
                float_T const timeMod = float_T(time - numberOfPeriods * deltaT);
                float_T const yMod = float_T(pos.y() + numberOfPeriods * deltaY);
                float_T const zMod = float_T(pos.z() + numberOfPeriods * deltaZ);

                float_T const x = float_T(phiPositive * pos.x() / UNIT_LENGTH);
                float_T const y = float_T(phiPositive * yMod / UNIT_LENGTH);
                float_T const z = float_T(zMod / UNIT_LENGTH);
                float_T const t = float_T(timeMod / UNIT_TIME);

                /* Calculating shortcuts for speeding up field calculation */
                float_T sinPhi;
                float_T cosPhi;
                pmacc::math::sincos(phiT, sinPhi, cosPhi);
                float_T const cscPhi = float_T(1.0) / sinPhi;
                float_T const sinPhi2 = math::sin(phiT / float_T(2.0));
                float_T const sin2Phi = math::sin(phiT * float_T(2.0));
                float_T const tanPhi2 = math::tan(phiT / float_T(2.0));

                float_T const sinPhi_2 = sinPhi * sinPhi;
                float_T const sinPhi_3 = sinPhi * sinPhi_2;
                float_T const sinPhi_4 = sinPhi_2 * sinPhi_2;

                float_T const sinPhi2_2 = sinPhi2 * sinPhi2;
                float_T const sinPhi2_4 = sinPhi2_2 * sinPhi2_2;
                float_T const tanPhi2_2 = tanPhi2 * tanPhi2;

                float_T const tauG2 = tauG * tauG;
                float_T const x2 = x * x;
                float_T const y2 = y * y;
                float_T const z2 = z * z;

                /* The "helpVar" variables decrease the nesting level of the evaluated expressions and
                 * thus help with formal code verification through manual code inspection.
                 */
                complex_T const helpVar1 = cspeed * om0 * tauG2 * sinPhi_4
                    - complex_T(0, 8) * sinPhi2_4 * sinPhi * (y * cosPhi + z * sinPhi);

                complex_T const helpVar2 = complex_T(0, 1) * rho0 - y * cosPhi - z * sinPhi;

                complex_T const helpVar3 = complex_T(0, float_T(-0.5)) * cscPhi
                    * (complex_T(0, -8) * om0 * y * (cspeed * t - z) * sinPhi2_2 * sinPhi_4
                           * (complex_T(0, 1) * rho0 - z * sinPhi)
                       - om0 * sinPhi_4 * sinPhi
                           * (-float_T(2.0) * z2 * rho0
                              - cspeed * cspeed
                                  * (k * tauG2 * x2 + float_T(2.0) * t * (t - complex_T(0, 1) * om0 * tauG2) * rho0)
                              + cspeed * (float_T(4.0) * t * z * rho0 - complex_T(0, 2) * om0 * tauG2 * z * rho0)
                              - complex_T(0, 2) * (cspeed * t - z) * (cspeed * (t - complex_T(0, 1) * om0 * tauG2) - z)
                                  * z * sinPhi)
                       + float_T(2.0) * y * cosPhi * sinPhi_2
                           * (complex_T(0, 4) * om0 * y * (cspeed * t - z) * sinPhi2_2 * sinPhi_2
                              + om0 * (cspeed * t - z)
                                  * (complex_T(0, 1) * cspeed * t + cspeed * om0 * tauG2 - complex_T(0, 1) * z)
                                  * sinPhi_3
                              - complex_T(0, 4) * sinPhi2_4
                                  * (cspeed * k * x2 - om0 * (y2 - float_T(4.0) * (cspeed * t - z) * z) * sinPhi))
                       - complex_T(0, 4) * sinPhi2_4
                           * (complex_T(0, -4) * om0 * y * (cspeed * t - z) * rho0 * cosPhi * sinPhi_2
                              + complex_T(0, 2)
                                  * (om0 * (y2 + float_T(2.0) * z2) * rho0
                                     - cspeed * z * (complex_T(0, 1) * k * x2 + float_T(2.0) * om0 * t * rho0))
                                  * sinPhi_3
                              - float_T(2.0) * om0 * z * (y2 - float_T(2.0) * (cspeed * t - z) * z) * sinPhi_4
                              + om0 * y2 * (cspeed * t - z) * sin2Phi * sin2Phi))
                    / (cspeed * helpVar2 * helpVar1);

                complex_T const helpVar4 = cspeed * om0 * tauG2
                    - complex_T(0, 8) * y * math::tan(float_T(PI / 2.0) - phiT) * cscPhi * cscPhi * sinPhi2_4
                    - complex_T(0, 2) * z * tanPhi2_2;

                complex_T const result
                    = (math::exp(helpVar3) * tauG * math::sqrt(cspeed * om0 * rho0 / helpVar2)) / math::sqrt(helpVar4);

                return result.get_real();
            }

            /** Calculate the Ey(r,t) field here
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations) for calculating
             *             the field */
            HDINLINE EField::float_T EField::calcTWTSEy(float3_64 const& pos, float_64 const time) const
            {
                /* The field function of Ey (polarization in pulse-front-tilt plane)
                 * is by definition identical to Ex (polarization normal to pulse-front-tilt plane)
                 */
                return calcTWTSEx(pos, time);
            }

        } /* namespace twtsfast */
    } /* namespace templates */
} /* namespace picongpu */
