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
#include "picongpu/fields/background/templates/twtsfast/BField.hpp"
#include "picongpu/fields/CellType.hpp"


namespace picongpu
{
    /** Load pre-defined background field */
    namespace templates
    {
        /** Traveling-wave Thomson scattering laser pulse */
        namespace twtsfast
        {
            HINLINE
            BField::BField(
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
                 * on host (see fieldBackground.param), this is no problem.
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
                if(phi < float_X(0.0))
                    phiPositive = float_X(-1.0);
            }

            template<>
            HDINLINE float3_X BField::getTWTSBfield_Normalized<DIM3>(
                pmacc::math::Vector<floatD_64, detail::numComponents> const& bFieldPositions_SI,
                float_64 const time) const
            {
                using PosVecVec = pmacc::math::Vector<float3_64, detail::numComponents>;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    for(uint32_t i = 0; i < simDim; ++i)
                    {
                        pos[k][i] = bFieldPositions_SI[k][i];
                    }
                }

                /* An example of intra-cell position offsets is the staggered Yee-grid.
                 *
                 * Calculate By-component with the intra-cell offset of a By-field
                 */
                float_64 const By_By = calcTWTSBy(pos[1], time);
                /* Calculate Bz-component the the intra-cell offset of a By-field */
                float_64 const Bz_By = calcTWTSBz_Ex(pos[1], time);
                /* Calculate By-component the the intra-cell offset of a Bz-field */
                float_64 const By_Bz = calcTWTSBy(pos[2], time);
                /* Calculate Bz-component the the intra-cell offset of a Bz-field */
                float_64 const Bz_Bz = calcTWTSBz_Ex(pos[2], time);
                /* Since we rotated all position vectors before calling calcTWTSBy and calcTWTSBz_Ex,
                 * we need to back-rotate the resulting B-field vector.
                 *
                 * RotationMatrix[-(PI/2+phi)].(By,Bz) for rotating back the field vectors.
                 */
                float_X sinPhi;
                float_X cosPhi;
                pmacc::math::sincos(phi, sinPhi, cosPhi);
                float_X const By_rot = -sinPhi * float_X(By_By) + cosPhi * float_X(Bz_By);
                float_X const Bz_rot = -cosPhi * float_X(By_Bz) - sinPhi * float_X(Bz_Bz);

                /* Finally, the B-field normalized to the peak amplitude. */
                return float3_X(0.0_X, By_rot, Bz_rot);
            }

            template<>
            HDINLINE float3_X BField::getTWTSBfield_Normalized_Ey<DIM3>(
                pmacc::math::Vector<floatD_64, detail::numComponents> const& bFieldPositions_SI,
                float_64 const time) const
            {
                using PosVecVec = pmacc::math::Vector<float3_64, detail::numComponents>;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    for(uint32_t i = 0; i < simDim; ++i)
                    {
                        pos[k][i] = bFieldPositions_SI[k][i];
                    }
                }

                /* Calculate Bz-component with the intra-cell offset of a By-field */
                float_64 const Bz_By = calcTWTSBz_Ey(pos[1], time);
                /* Calculate Bz-component with the intra-cell offset of a Bz-field */
                float_64 const Bz_Bz = calcTWTSBz_Ey(pos[2], time);
                /* Since we rotated all position vectors before calling calcTWTSBz_Ey,
                 * we need to back-rotate the resulting B-field vector.
                 *
                 * RotationMatrix[-(PI/2+phi)].(By,Bz) for rotating back the field-vectors.
                 */
                float_X sinPhi;
                float_X cosPhi;
                pmacc::math::sincos(phi, sinPhi, cosPhi);
                float_X const By_rot = +cosPhi * float_X(Bz_By);
                float_X const Bz_rot = -sinPhi * float_X(Bz_Bz);

                /* Finally, the B-field normalized to the peak amplitude. */
                return float3_X(float_X(calcTWTSBx(pos[0], time)), By_rot, Bz_rot);
            }

            template<>
            HDINLINE float3_X BField::getTWTSBfield_Normalized<DIM2>(
                pmacc::math::Vector<floatD_64, detail::numComponents> const& bFieldPositions_SI,
                float_64 const time) const
            {
                using PosVecVec = pmacc::math::Vector<float3_64, detail::numComponents>;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    /* 2D (y,z) vectors are mapped on 3D (x,y,z) vectors. */
                    for(uint32_t i = 0; i < DIM2; ++i)
                    {
                        pos[k][i + 1] = bFieldPositions_SI[k][i];
                    }
                }

                /* General background comment for the rest of this function:
                 *
                 * Corresponding position vector for the field components in 2D simulations.
                 *  3D     3D vectors in 2D space (x, y)
                 *  x -->  z (Meaning: In 2D-sim, insert cell-coordinate x
                 *            into TWTS field function coordinate z.)
                 *  y -->  y
                 *  z --> -x (Since z=0 for 2D, we use the existing
                 *            3D TWTS-field-function and set x = -0)
                 *  The transformed 3D coordinates are used to calculate the field components.
                 *  Ex --> Ez (Meaning: Calculate Ex-component of existing 3D TWTS-field (calcTWTSEx) using
                 *             transformed position vectors to obtain the corresponding Ez-component in 2D.
                 *             Note: Swapping field component coordinates also alters the
                 *                   intra-cell position offset.)
                 *  By --> By
                 *  Bz --> -Bx (Yes, the sign is necessary.)
                 *
                 * An example of intra-cell position offsets is the staggered Yee-grid.
                 *
                 * This procedure is analogous to 3D case, but replace By --> By and Bz --> -Bx. Hence the
                 * grid cell offset for Bx has to be used instead of Bz. Mind the "-"-sign.
                 */

                /* Calculate By-component with the intra-cell offset of a By-field */
                float_64 const By_By = calcTWTSBy(pos[1], time);
                /* Calculate Bx-component with the intra-cell offset of a By-field */
                float_64 const Bx_By = -calcTWTSBz_Ex(pos[1], time);
                /* Calculate By-component with the intra-cell offset of a Bx-field */
                float_64 const By_Bx = calcTWTSBy(pos[0], time);
                /* Calculate Bx-component with the intra-cell offset of a Bx-field */
                float_64 const Bx_Bx = -calcTWTSBz_Ex(pos[0], time);
                /* Since we rotated all position vectors before calling calcTWTSBy and calcTWTSBz_Ex, we
                 * need to back-rotate the resulting B-field vector. Now the rotation is done
                 * analogously in the (y,x)-plane. (Reverse of the position vector transformation.)
                 *
                 * RotationMatrix[-(PI / 2+phi)].(By,Bx) for rotating back the field vectors.
                 */
                float_X sinPhi;
                float_X cosPhi;
                pmacc::math::sincos(phi, sinPhi, cosPhi);
                float_X const By_rot = -sinPhi * float_X(By_By) + cosPhi * float_X(Bx_By);
                float_X const Bx_rot = -cosPhi * float_X(By_Bx) - sinPhi * float_X(Bx_Bx);

                /* Finally, the B-field normalized to the peak amplitude. */
                return float3_X(Bx_rot, By_rot, 0.0_X);
            }

            template<>
            HDINLINE float3_X BField::getTWTSBfield_Normalized_Ey<DIM2>(
                pmacc::math::Vector<floatD_64, detail::numComponents> const& bFieldPositions_SI,
                float_64 const time) const
            {
                using PosVecVec = pmacc::math::Vector<float3_64, detail::numComponents>;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    /* The 2D output of getFieldPositions_SI only returns
                     * the y- and z-component of a 3D vector.
                     */
                    for(uint32_t i = 0; i < DIM2; ++i)
                    {
                        pos[k][i + 1] = bFieldPositions_SI[k][i];
                    }
                }

                /* General background comment for the rest of this function:
                 *
                 * Corresponding position vector for the field components in 2D simulations.
                 *  3D     3D vectors in 2D space (x, y)
                 *  x -->  z (Meaning: In 2D-sim, insert cell-coordinate x
                 *            into TWTS field function coordinate z.)
                 *  y -->  y
                 *  z --> -x (Since z=0 for 2D, we use the existing
                 *            3D TWTS-field-function and set x = -0)
                 *  Ex --> Ez (Meaning: Calculate Ex-component of existing 3D TWTS-field to obtain
                 *             corresponding Ez-component in 2D.
                 *             Note: the intra-cell position offset due to the staggered grid for Ez.)
                 *  By --> By
                 *  Bz --> -Bx (Yes, the sign is necessary.)
                 *
                 * This procedure is analogous to 3D case, but replace By --> By and Bz --> -Bx. Hence the
                 * grid cell offset for Bx has to be used instead of Bz. Mind the -sign.
                 */

                /* Calculate Bx-component with the intra-cell offset of a By-field */
                float_64 const Bx_By = -calcTWTSBz_Ex(pos[1], time);
                /* Calculate Bx-component with the intra-cell offset of a Bx-field */
                float_64 const Bx_Bx = -calcTWTSBz_Ex(pos[0], time);

                /* Since we rotated all position vectors before calling calcTWTSBz_Ex, we
                 * need to back-rotate the resulting B-field vector. Now the rotation is done
                 * analogously in the (y,x)-plane. (Reverse of the position vector transformation.)
                 *
                 * RotationMatrix[-(PI / 2+phi)].(By,Bx)
                 * for rotating back the field-vectors.
                 */
                float_X sinPhi;
                float_X cosPhi;
                pmacc::math::sincos(phi, sinPhi, cosPhi);
                float_X const By_rot = +cosPhi * float_X(Bx_By);
                float_X const Bx_rot = -sinPhi * float_X(Bx_Bx);

                /* Finally, the B-field normalized to the peak amplitude. */
                return float3_X(Bx_rot, By_rot, float_X(calcTWTSBx(pos[2], time)));
            }

            HDINLINE
            float3_X BField::operator()(DataSpace<simDim> const& cellIdx, uint32_t const currentStep) const
            {
                float_64 const time_SI = float_64(currentStep) * dt - tdelay;
                traits::FieldPosition<fields::CellType, FieldB> const fieldPosB;

                pmacc::math::Vector<floatD_64, detail::numComponents> const bFieldPositions_SI
                    = detail::getFieldPositions_SI(cellIdx, halfSimSize, fieldPosB(), unit_length, focus_y_SI, phi);
                /* Single TWTS-Pulse */
                switch(pol)
                {
                case LINEAR_X:
                    return getTWTSBfield_Normalized<simDim>(bFieldPositions_SI, time_SI);

                case LINEAR_YZ:
                    return getTWTSBfield_Normalized_Ey<simDim>(bFieldPositions_SI, time_SI);
                }
                return getTWTSBfield_Normalized<simDim>(bFieldPositions_SI,
                                                        time_SI); // defensive default
            }

            /** Calculate the By(r,t) field here
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations)
             *             for calculating the field */
            HDINLINE
            BField::float_T BField::calcTWTSBy(float3_64 const& pos, float_64 const time) const
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
                 * For beta0=1.0, this is equivalent to our standard definition. Question: Why is the
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
                 * float_T const eta = float_T(PI/2) - (phiReal - alphaTilt);
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
                float_T const secPhi2 = float_T(1.0) / math::cos(phiT / float_T(2.0));
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
                const complex_T helpVar1 = cspeed * om0 * tauG2 * sinPhi_4
                    - complex_T(0, 8) * sinPhi2_4 * sinPhi * (y * cosPhi + z * sinPhi);

                const complex_T helpVar2 = complex_T(0, 1) * rho0 - y * cosPhi - z * sinPhi;

                const complex_T helpVar3
                    = (complex_T(0, float_T(-0.5)) * cscPhi
                       * (complex_T(0, -8) * om0 * y * (cspeed * t - z) * sinPhi2_2 * sinPhi_4
                              * (complex_T(0, 1) * rho0 - z * sinPhi)
                          - om0 * sinPhi_4 * sinPhi
                              * (-float_t(2.0) * z2 * rho0
                                 - cspeed * cspeed
                                     * (k * tauG2 * x2 + float_t(2.0) * t * (t - complex_T(0, 1) * om0 * tauG2) * rho0)
                                 + cspeed * (4 * t * z * rho0 - complex_T(0, 2) * om0 * tauG2 * z * rho0)
                                 - complex_T(0, 2) * (cspeed * t - z)
                                     * (cspeed * (t - complex_T(0, 1) * om0 * tauG2) - z) * z * sinPhi)
                          + y * sinPhi
                              * (complex_T(0, 4) * om0 * y * (cspeed * t - z) * sinPhi2_2 * sinPhi_2
                                 + om0 * (cspeed * t - z)
                                     * (complex_T(0, 1) * cspeed * t + cspeed * om0 * tauG2 - complex_T(0, 1) * z)
                                     * sinPhi_3
                                 - complex_T(0, 4) * sinPhi2_4
                                     * (cspeed * k * x2 - om0 * (y2 - float_T(4.0) * (cspeed * t - z) * z) * sinPhi))
                              * sin2Phi
                          - complex_T(0, 4) * sinPhi2_4
                              * (complex_T(0, -4) * om0 * y * (cspeed * t - z) * rho0 * cosPhi * sinPhi_2
                                 + complex_T(0, 2)
                                     * (om0 * (y2 + float_T(2.0) * z2) * rho0
                                        - cspeed * z * (complex_T(0, 1) * k * x2 + float_T(2.0) * om0 * t * rho0))
                                     * sinPhi_3
                                 - float_T(2.0) * om0 * z * (y2 - float_T(2.0) * (cspeed * t - z) * z) * sinPhi_4
                                 + om0 * y2 * (cspeed * t - z) * sin2Phi * sin2Phi)))
                    / (cspeed * helpVar2 * helpVar1);

                complex_T const helpVar4 = cspeed * om0 * tauG * tauG
                    - complex_T(0, 8) * y * math::tan(float_T(PI / 2.0) - phiT) * cscPhi * cscPhi * sinPhi2_4
                    - complex_T(0, 2) * z * tanPhi2_2;

                complex_T const result
                    = (math::exp(helpVar3) * tauG * secPhi2 * secPhi2
                       * (complex_T(0, 2) * cspeed * t + cspeed * om0 * tauG2 - complex_T(0, 4) * z
                          + cspeed * (complex_T(0, 2) * t + om0 * tauG2) * cosPhi + complex_T(0, 2) * y * tanPhi2)
                       * math::sqrt(cspeed * om0 * rho0 / helpVar2))
                    / (float_T(2.0) * cspeed * math::pow(helpVar4, float_T(1.5)));

                return result.get_real() / UNIT_SPEED;
            }

            /** Calculate the Bz(r,t) field
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations)
             *             for calculating the field */
            HDINLINE
            BField::float_T BField::calcTWTSBz_Ex(float3_64 const& pos, float_64 const time) const
            {
                using complex_T = pmacc::math::Complex<float_T>;

                /* propagation speed of overlap normalized to the speed of light [Default: beta0=1.0] */
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
                 * For beta0=1.0, this is equivalent to our standard definition. Question: Why is the
                 * local "phi_T" not equal in value to the object member "phiReal" or "phi"?
                 * Because the standard TWTS pulse is defined for beta0 = 1.0 and in the coordinate-system
                 * of the TWTS model phi is responsible for pulse front tilt and dispersion only. Hence
                 * the dispersion will (although physically correct) be slightly off the ideal TWTS
                 * pulse for beta0 != 1.0. This only shows that this TWTS pulse is primarily designed for
                 * scenarios close to beta0 = 1.
                 */
                float_T const phiT = float_T(2.0) * alphaTilt;

                /* Angle between the laser pulse front and the y-axis.
                 * Not used, but remains in code for documentation purposes.
                 * float_T const eta = float_T(float_T(PI / 2)) - (phiReal - alphaTilt);
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
                float_T const secPhi2 = float_T(1.0) / math::cos(phiT / float_T(2.0));
                float_T const sinPhi2 = math::sin(phiT / float_T(2.0));
                float_T const tanPhi2 = math::tan(phiT / float_T(2.0));

                float_T const cscPhi_3 = cscPhi * cscPhi * cscPhi;

                float_T const sinPhi2_2 = sinPhi2 * sinPhi2;
                float_T const sinPhi2_4 = sinPhi2_2 * sinPhi2_2;
                float_T const tanPhi2_2 = tanPhi2 * tanPhi2;
                float_T const secPhi2_2 = secPhi2 * secPhi2;

                float_T const tanPI2_phi = math::tan(float_T(PI / 2.0) - phiT);

                float_T const tauG2 = tauG * tauG;
                float_T const om02 = om0 * om0;
                float_T const x2 = x * x;
                float_T const y2 = y * y;
                float_T const z2 = z * z;

                /* The "helpVar" variables decrease the nesting level of the evaluated expressions and
                 * thus help with formal code verification through manual code inspection.
                 */
                const complex_T helpVar1 = cspeed * om0 * tauG2 - complex_T(0, 1) * y * cosPhi * secPhi2_2 * tanPhi2
                    - complex_T(0, 2) * z * tanPhi2_2;
                const complex_T helpVar2 = complex_T(0, 1) * cspeed * rho0 - cspeed * y * cosPhi - cspeed * z * sinPhi;
                const complex_T helpVar3 = rho0 + complex_T(0, 1) * y * cosPhi + complex_T(0, 1) * z * sinPhi;
                const complex_T helpVar4 = complex_T(0, 1) * rho0 - y * cosPhi - z * sinPhi;
                const complex_T helpVar5 = -z - y * tanPI2_phi + complex_T(0, 1) * rho0 * cscPhi;
                const complex_T helpVar6
                    = -cspeed * z - cspeed * y * tanPI2_phi + complex_T(0, 1) * cspeed * rho0 * cscPhi;
                const complex_T helpVar7 = complex_T(0, 1) * cspeed * rho0 - cspeed * y * cosPhi - cspeed * z * sinPhi;

                const complex_T helpVar8
                    = (om0 * y * rho0 * secPhi2_2 * secPhi2_2 / helpVar6
                       + (om0 * y * tanPI2_phi
                          * (cspeed * om0 * tauG2
                             + float_T(8.0) * (complex_T(0, 2) * y + rho0) * cscPhi_3 * sinPhi2_4))
                           / (cspeed * helpVar5)
                       + om02 * tauG2 * z * sinPhi / helpVar4 - float_T(2.0) * k * x2 / helpVar3
                       - om02 * tauG2 * rho0 / helpVar3
                       + complex_T(0, 1) * om0 * y2 * cosPhi * cosPhi * secPhi2_2 * tanPhi2 / helpVar2
                       + complex_T(0, 4) * om0 * y * z * tanPhi2_2 / helpVar2
                       - float_T(2.0) * om0 * z * rho0 * tanPhi2_2 / helpVar2
                       - complex_T(0, 2) * om0 * z2 * sinPhi * tanPhi2_2 / helpVar2
                       - (om0
                          * math::pow(
                              float_T(2.0) * cspeed * t - complex_T(0, 1) * cspeed * om0 * tauG2 - float_T(2.0) * z
                                  + float_T(8.0) * y * cscPhi_3 * sinPhi2_4 - float_T(2.0) * z * tanPhi2_2,
                              float_T(2.0)))
                           / (cspeed * helpVar1))
                    / float_T(4.0);

                const complex_T helpVar9 = cspeed * om0 * tauG2 - complex_T(0, 1) * y * cosPhi * secPhi2_2 * tanPhi2
                    - complex_T(0, 2) * z * tanPhi2_2;

                const complex_T result = float_T(phiPositive)
                    * (complex_T(0, 2) * math::exp(helpVar8) * tauG * tanPhi2 * (cspeed * t - z + y * tanPhi2)
                       * math::sqrt(om0 * rho0 / helpVar7))
                    / math::pow(helpVar9, float_T(1.5));

                return result.get_real() / UNIT_SPEED;
            }

            /** Calculate the Bx(r,t) field
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations)
             *             for calculating the field */
            HDINLINE
            BField::float_T BField::calcTWTSBx(float3_64 const& pos, float_64 const time) const
            {
                /* The Bx-field for the Ey-field is the same as
                 * for the By-field for the Ex-field except for the sign.
                 */
                return -calcTWTSBy(pos, time);
            }

            /** Calculate the Bz(r,t) field
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations)
             *             for calculating the field */
            HDINLINE
            BField::float_T BField::calcTWTSBz_Ey(float3_64 const& pos, float_64 const time) const
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
                float_T cosPhiReal;
                float_T sinPhiReal;
                pmacc::math::sincos(phiReal, sinPhiReal, cosPhiReal);
                float_T const alphaTilt = math::atan2(float_T(1.0) - beta0 * cosPhiReal, beta0 * sinPhiReal);
                /* Definition of the laser pulse front tilt angle for the laser field below.
                 *
                 * For beta0=1.0, this is equivalent to our standard definition. Question: Why is the
                 * local "phi_T" not equal in value to the object member "phiReal" or "phi"?
                 * Because the standard TWTS pulse is defined for beta0 = 1.0 and in the coordinate-system
                 * of the TWTS model phi is responsible for pulse front tilt and dispersion only. Hence
                 * the dispersion will (although physically correct) be slightly off the ideal TWTS
                 * pulse for beta0 != 1.0. This only shows that this TWTS pulse is primarily designed for
                 * scenarios close to beta0 = 1.
                 */
                float_T const phiT = float_T(2.0) * alphaTilt;

                /* Angle between the laser pulse front and the y-axis.
                 * Not used, but remains in code for documentation purposes.
                 * float_T const eta = float_T(float_T(PI / 2)) - (phiReal - alphaTilt);
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

                /* Shortcuts for speeding up the field calculation. */
                float_T sinPhi;
                float_T cosPhi;
                pmacc::math::sincos(phiT, sinPhi, cosPhi);
                float_T const sin2Phi = math::sin(phiT * float_T(2.0));
                float_T const sinPhi2 = math::sin(phiT / float_T(2.0));
                float_T const tanPhi2 = math::tan(phiT / float_T(2.0));

                float_T const cscPhi = float_T(1.0) / sinPhi;
                float_T const tanPI2_phi = math::tan(float_T(PI / 2.0) - phiT);

                float_T const sinPhi_2 = sinPhi * sinPhi;
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
                const complex_T helpVar1 = cspeed * om0 * tauG2 * sinPhi_4
                    - complex_T(0, 8) * sinPhi2_4 * sinPhi * (y * cosPhi + z * sinPhi);

                const complex_T helpVar2 = complex_T(0, 1) * rho0 - y * cosPhi - z * sinPhi;

                const complex_T helpVar3
                    = (complex_T(0, float_T(-0.5)) * cscPhi
                       * (complex_T(0, -8) * om0 * y * (cspeed * t - z) * sinPhi2_2 * sinPhi_4
                              * (complex_T(0, 1) * rho0 - z * sinPhi)
                          - om0 * sinPhi * sinPhi_4
                              * (float_T(-2.0) * z2 * rho0
                                 - cspeed * cspeed
                                     * (k * tauG2 * x2 + float_T(2.0) * t * (t - complex_T(0, 1) * om0 * tauG2) * rho0)
                                 + cspeed * (float_T(4.0) * t * z * rho0 - complex_T(0, 2) * om0 * tauG2 * z * rho0)
                                 - complex_T(0, 2) * (cspeed * t - z)
                                     * (cspeed * (t - complex_T(0, 1) * om0 * tauG2) - z) * z * sinPhi)
                          + float_T(2.0) * y * cosPhi * sinPhi_2
                              * (complex_T(0, 4) * om0 * y * (cspeed * t - z) * sinPhi2_2 * sinPhi_2
                                 + om0 * (cspeed * t - z)
                                     * (complex_T(0, 1) * cspeed * t + cspeed * om0 * tauG2 - complex_T(0, 1) * z)
                                     * sinPhi_2 * sinPhi
                                 - complex_T(0, 4) * sinPhi2_4
                                     * (cspeed * k * x2 - om0 * (y2 - float_T(4.0) * (cspeed * t - z) * z) * sinPhi))
                          - complex_T(0, 4) * sinPhi2_4
                              * (complex_T(0, -4) * om0 * y * (cspeed * t - z) * rho0 * cosPhi * sinPhi_2
                                 + complex_T(0, 2)
                                     * (om0 * (y2 + float_T(2.0) * z2) * rho0
                                        - cspeed * z * (complex_T(0, 1) * k * x2 + float_T(2.0) * om0 * t * rho0))
                                     * sinPhi_2 * sinPhi
                                 - float_T(2.0) * om0 * z * (y2 - float_T(2.0) * (cspeed * t - z) * z) * sinPhi_4
                                 + om0 * y2 * (cspeed * t - z) * sin2Phi * sin2Phi))
                       /* The "round-trip" conversion in the line below fixes a gross accuracy bug
                        * in floating-point arithmetics, when float_T is set to float_X.
                        */
                       )
                    * complex_T(1.0 / complex_64(cspeed * helpVar2 * helpVar1));

                const complex_T helpVar4 = cspeed * om0 * rho0
                    * (cspeed * om0 * tauG2 - complex_T(0, 8) * y * tanPI2_phi * cscPhi * cscPhi * sinPhi2_4
                       - complex_T(0, 2) * z * tanPhi2_2);

                const complex_T result = float_T(-1.0)
                    * (cspeed * math::exp(helpVar3) * k * tauG * x * rho0
                       * math::pow(float_T(1.0) / helpVar2, float_T(1.5)))
                    / math::sqrt(helpVar4);

                return result.get_real() / UNIT_SPEED;
            }

        } /* namespace twtsfast */
    } /* namespace templates */
} /* namespace picongpu */
