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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/CellType.hpp"
#include "picongpu/fields/background/templates/TWTS/BField.hpp"
#include "picongpu/fields/background/templates/TWTS/GetInitialTimeDelay_SI.tpp"
#include "picongpu/fields/background/templates/TWTS/RotateField.tpp"
#include "picongpu/fields/background/templates/TWTS/getFieldPositions_SI.tpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/math/Complex.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/types.hpp>


namespace picongpu
{
    /** Load pre-defined background field */
    namespace templates
    {
        /** Traveling-wave Thomson scattering laser pulse */
        namespace twts
        {
            HINLINE
            BField::BField(
                const float_64 focus_y_SI,
                const float_64 wavelength_SI,
                const float_64 pulselength_SI,
                const float_64 w_x_SI,
                const float_64 w_y_SI,
                const float_T phi,
                const float_T beta_0,
                const float_64 tdelay_user_SI,
                const bool auto_tdelay,
                const PolarizationType pol)
                : focus_y_SI(focus_y_SI)
                , wavelength_SI(wavelength_SI)
                , pulselength_SI(pulselength_SI)
                , w_x_SI(w_x_SI)
                , w_y_SI(w_y_SI)
                , phi(phi)
                , beta_0(beta_0)
                , tdelay_user_SI(tdelay_user_SI)
                , dt(SI::DELTA_T_SI)
                , unit_length(UNIT_LENGTH)
                , auto_tdelay(auto_tdelay)
                , pol(pol)
                , phiPositive(float_T(1.0))
            {
                /* Note: Enviroment-objects cannot be instantiated on CUDA GPU device. Since this is done
                 * on host (see fieldBackground.param), this is no problem.
                 */
                const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                halfSimSize = subGrid.getGlobalDomain().size / 2;
                tdelay = detail::getInitialTimeDelay_SI(
                    auto_tdelay,
                    tdelay_user_SI,
                    halfSimSize,
                    pulselength_SI,
                    focus_y_SI,
                    phi,
                    beta_0);
                if(phi < float_T(0.0))
                    phiPositive = float_T(-1.0);
            }

            template<>
            HDINLINE float3_X BField::getTWTSBfield_Normalized<DIM3>(
                const pmacc::math::Vector<floatD_64, detail::numComponents>& bFieldPositions_SI,
                const float_64 time) const
            {
                typedef pmacc::math::Vector<float3_64, detail::numComponents> PosVecVec;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    for(uint32_t i = 0; i < simDim; ++i)
                        pos[k][i] = bFieldPositions_SI[k][i];
                }

                /* An example of intra-cell position offsets is the staggered Yee-grid.
                 *
                 * Calculate By-component with the intra-cell offset of a By-field
                 */
                const float_64 By_By = calcTWTSBy(pos[1], time);
                /* Calculate Bz-component the the intra-cell offset of a By-field */
                const float_64 Bz_By = calcTWTSBz_Ex(pos[1], time);
                /* Calculate By-component the the intra-cell offset of a Bz-field */
                const float_64 By_Bz = calcTWTSBy(pos[2], time);
                /* Calculate Bz-component the the intra-cell offset of a Bz-field */
                const float_64 Bz_Bz = calcTWTSBz_Ex(pos[2], time);
                /* Since we rotated all position vectors before calling calcTWTSBy and calcTWTSBz_Ex,
                 * we need to back-rotate the resulting B-field vector.
                 *
                 * RotationMatrix[-(PI/2+phi)].(By,Bz) for rotating back the field vectors.
                 */
                float_T sinPhi;
                float_T cosPhi;
                pmacc::math::sincos(phi, sinPhi, cosPhi);
                const float_64 By_rot = -sinPhi * By_By + cosPhi * Bz_By;
                const float_64 Bz_rot = -cosPhi * By_Bz - sinPhi * Bz_Bz;

                /* Finally, the B-field normalized to the peak amplitude. */
                return float3_X(float_X(0.0), float_X(By_rot), float_X(Bz_rot));
            }

            template<>
            HDINLINE float3_X BField::getTWTSBfield_Normalized_Ey<DIM3>(
                const pmacc::math::Vector<floatD_64, detail::numComponents>& bFieldPositions_SI,
                const float_64 time) const
            {
                typedef pmacc::math::Vector<float3_64, detail::numComponents> PosVecVec;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    for(uint32_t i = 0; i < simDim; ++i)
                        pos[k][i] = bFieldPositions_SI[k][i];
                }

                /* Calculate Bz-component with the intra-cell offset of a By-field */
                const float_64 Bz_By = calcTWTSBz_Ey(pos[1], time);
                /* Calculate Bz-component with the intra-cell offset of a Bz-field */
                const float_64 Bz_Bz = calcTWTSBz_Ey(pos[2], time);
                /* Since we rotated all position vectors before calling calcTWTSBz_Ey,
                 * we need to back-rotate the resulting B-field vector.
                 *
                 * RotationMatrix[-(PI/2+phi)].(By,Bz) for rotating back the field-vectors.
                 */
                float_T sinPhi;
                float_T cosPhi;
                pmacc::math::sincos(phi, sinPhi, cosPhi);
                const float_64 By_rot = +cosPhi * Bz_By;
                const float_64 Bz_rot = -sinPhi * Bz_Bz;

                /* Finally, the B-field normalized to the peak amplitude. */
                return float3_X(float_X(calcTWTSBx(pos[0], time)), float_X(By_rot), float_X(Bz_rot));
            }

            template<>
            HDINLINE float3_X BField::getTWTSBfield_Normalized<DIM2>(
                const pmacc::math::Vector<floatD_64, detail::numComponents>& bFieldPositions_SI,
                const float_64 time) const
            {
                typedef pmacc::math::Vector<float3_64, detail::numComponents> PosVecVec;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    /* 2D (y,z) vectors are mapped on 3D (x,y,z) vectors. */
                    for(uint32_t i = 0; i < DIM2; ++i)
                        pos[k][i + 1] = bFieldPositions_SI[k][i];
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
                const float_64 By_By = calcTWTSBy(pos[1], time);
                /* Calculate Bx-component with the intra-cell offset of a By-field */
                const float_64 Bx_By = -calcTWTSBz_Ex(pos[1], time);
                /* Calculate By-component with the intra-cell offset of a Bx-field */
                const float_64 By_Bx = calcTWTSBy(pos[0], time);
                /* Calculate Bx-component with the intra-cell offset of a Bx-field */
                const float_64 Bx_Bx = -calcTWTSBz_Ex(pos[0], time);
                /* Since we rotated all position vectors before calling calcTWTSBy and calcTWTSBz_Ex, we
                 * need to back-rotate the resulting B-field vector. Now the rotation is done
                 * analogously in the (y,x)-plane. (Reverse of the position vector transformation.)
                 *
                 * RotationMatrix[-(PI / 2+phi)].(By,Bx) for rotating back the field vectors.
                 */
                float_X sinPhi;
                float_X cosPhi;
                pmacc::math::sincos(phi, sinPhi, cosPhi);
                const float_64 By_rot = -sinPhi * By_By + cosPhi * Bx_By;
                const float_64 Bx_rot = -cosPhi * By_Bx - sinPhi * Bx_Bx;

                /* Finally, the B-field normalized to the peak amplitude. */
                return float3_X(float_X(Bx_rot), float_X(By_rot), float_X(0.0));
            }

            template<>
            HDINLINE float3_X BField::getTWTSBfield_Normalized_Ey<DIM2>(
                const pmacc::math::Vector<floatD_64, detail::numComponents>& bFieldPositions_SI,
                const float_64 time) const
            {
                typedef pmacc::math::Vector<float3_64, detail::numComponents> PosVecVec;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    /* The 2D output of getFieldPositions_SI only returns
                     * the y- and z-component of a 3D vector.
                     */
                    for(uint32_t i = 0; i < DIM2; ++i)
                        pos[k][i + 1] = bFieldPositions_SI[k][i];
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
                const float_64 Bx_By = -calcTWTSBz_Ex(pos[1], time);
                /* Calculate Bx-component with the intra-cell offset of a Bx-field */
                const float_64 Bx_Bx = -calcTWTSBz_Ex(pos[0], time);

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
                const float_64 By_rot = +cosPhi * Bx_By;
                const float_64 Bx_rot = -sinPhi * Bx_Bx;

                /* Finally, the B-field normalized to the peak amplitude. */
                return float3_X(float_X(Bx_rot), float_X(By_rot), float_X(calcTWTSBx(pos[2], time)));
            }

            HDINLINE float3_X BField::operator()(const DataSpace<simDim>& cellIdx, const uint32_t currentStep) const
            {
                const float_64 time_SI = float_64(currentStep) * dt - tdelay;
                const traits::FieldPosition<fields::CellType, FieldB> fieldPosB;

                const pmacc::math::Vector<floatD_64, detail::numComponents> bFieldPositions_SI
                    = detail::getFieldPositions_SI(cellIdx, halfSimSize, fieldPosB(), unit_length, focus_y_SI, phi);
                /* Single TWTS-Pulse */
                switch(pol)
                {
                case LINEAR_X:
                    return getTWTSBfield_Normalized<simDim>(bFieldPositions_SI, time_SI);

                case LINEAR_YZ:
                    return getTWTSBfield_Normalized_Ey<simDim>(bFieldPositions_SI, time_SI);
                }
                return getTWTSBfield_Normalized<simDim>(bFieldPositions_SI, time_SI); // defensive default
            }

            /** Calculate the By(r,t) field here
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations)
             *             for calculating the field */
            HDINLINE BField::float_T BField::calcTWTSBy(const float3_64& pos, const float_64 time) const
            {
                using complex_T = pmacc::math::Complex<float_T>;
                using complex_64 = pmacc::math::Complex<float_64>;

                /* If phi < 0 the formulas below are not directly applicable.
                 * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
                 * z-axis of the coordinate system in this function.
                 */
                const float_T phiReal = float_T(math::abs(phi));
                float_T sinPhiReal;
                float_T cosPhiReal;
                pmacc::math::sincos(phiReal, sinPhiReal, cosPhiReal);
                const float_T alphaTilt = math::atan2(float_T(1.0) - beta_0 * cosPhiReal, beta_0 * sinPhiReal);
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
                const float_T phiT = float_T(2.0) * alphaTilt;

                /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for
                 * documentation purposes.
                 * const float_T eta = float_T(PI/2) - (phiReal - alphaTilt);
                 */

                const float_T cspeed = float_T(SI::SPEED_OF_LIGHT_SI / UNIT_SPEED);
                const float_T lambda0 = float_T(wavelength_SI / UNIT_LENGTH);
                const float_T om0 = float_T(2.0 * PI * cspeed / lambda0);
                /* factor 2  in tauG arises from definition convention in laser formula */
                const float_T tauG = float_T(pulselength_SI * 2.0 / UNIT_TIME);
                /* w0 is wx here --> w0 could be replaced by wx */
                const float_T w0 = float_T(w_x_SI / UNIT_LENGTH);
                const float_T rho0 = float_T(PI * w0 * w0 / lambda0);
                /* wy is width of TWTS pulse */
                const float_T wy = float_T(w_y_SI / UNIT_LENGTH);
                const float_T k = float_T(2.0 * PI / lambda0);

                /* In order to calculate in single-precision and in order to account for errors in
                 * the approximations far from the coordinate origin, we use the wavelength-periodicity and
                 * the known propagation direction for realizing the laser pulse using relative coordinates
                 * (i.e. from a finite coordinate range) only. All these quantities have to be calculated
                 * in double precision.
                 */
                float_64 sinPhiVal;
                float_64 cosPhiVal;
                pmacc::math::sincos(precisionCast<float_64>(phi), sinPhiVal, cosPhiVal);
                const float_64 tanAlpha = (float_64(1.0) - beta_0 * cosPhiVal) / (beta_0 * sinPhiVal);
                const float_64 tanFocalLine = math::tan(PI / float_64(2.0) - phi);
                const float_64 deltaT
                    = wavelength_SI / SI::SPEED_OF_LIGHT_SI * (float_64(1.0) + tanAlpha / tanFocalLine);
                const float_64 deltaY = wavelength_SI / tanFocalLine;
                const float_64 deltaZ = -wavelength_SI;
                const float_64 numberOfPeriods = math::floor(time / deltaT);
                const float_T timeMod = float_T(time - numberOfPeriods * deltaT);
                const float_T yMod = float_T(pos.y() + numberOfPeriods * deltaY);
                const float_T zMod = float_T(pos.z() + numberOfPeriods * deltaZ);

                /* Find out the envelope factor along the (long) TWTS pulse width (y-axis)
                 * according to a Tukey-window.
                 * This is only correct if the transition region size deltawy = wy * alpha / 2
                 * is very much larger than the characteristic size of diffraction in the simulation.
                 * It is useful to compare the actual TWTS propagation distance (within the simulation volume)
                 * with the corresponding "Rayleigh length" PI * deltawy * deltawy / lambda0.
                 */

                const float_T x = float_T(phiPositive * pos.x() / UNIT_LENGTH);
                const float_T y = float_T(phiPositive * yMod / UNIT_LENGTH);
                const float_T z = float_T(zMod / UNIT_LENGTH);
                const float_T t = float_T(timeMod / UNIT_TIME);

                /* This makes the pulse super-gaussian (to the power of 8) and removes the previous purely gaussian
                 * dependency. This is a hack), which can only work close to the center of the Rayleigh length rho0,
                 * because it does not include eventual phase-evolution, due to super-gaussian focusing instead of
                 * gaussian focusing.
                 */

                /* Calculating shortcuts for speeding up field calculation */
                float_T sinPhi;
                float_T cosPhi;
                pmacc::math::sincos(phiT, sinPhi, cosPhi);
                const float_t phiThalf = phiT / float_T(2.0);
                float_T sinPhi2;
                float_T cosPhi2;
                pmacc::math::sincos(phiThalf, sinPhi2, cosPhi2);

                const float_T cscPhi = float_T(1.0) / sinPhi;
                const float_T secPhi2 = float_T(1.0) / cosPhi2;
                const float_T sin2Phi = math::sin(phiT * float_T(2.0));
                const float_T tanPhi2 = math::tan(phiThalf);

                const float_T sinPhi_2 = sinPhi * sinPhi;
                const float_T sinPhi_3 = sinPhi * sinPhi_2;
                const float_T sinPhi_4 = sinPhi_2 * sinPhi_2;

                const float_T sinPhi2_2 = sinPhi2 * sinPhi2;
                const float_T sinPhi2_4 = sinPhi2_2 * sinPhi2_2;
                const float_T tanPhi2_2 = tanPhi2 * tanPhi2;

                const float_T tauG2 = tauG * tauG;
                const float_T x2 = x * x;
                const float_T y2 = y * y;
                const float_T z2 = z * z;

                const float_T c_t = cspeed * t;
                const float_T c_Om0 = cspeed * om0;
                const float_T om0_tauG2 = om0 * tauG2;
                const float_T z_sinPhi = z * sinPhi;

                /* The "helpVar" variables decrease the nesting level of the evaluated expressions and
                 * thus help with formal code verification through manual code inspection.
                 */
                const complex_T helpVar1
                    = c_Om0 * tauG2 * sinPhi_4 - complex_T(0, 8) * (sinPhi2_4 * sinPhi * (y * cosPhi + z_sinPhi));

                const complex_T helpVar2 = complex_T(0, 1) * rho0 - y * cosPhi - z_sinPhi;

                const complex_T helpVar3
                    = (complex_T(0, float_T(-0.5)) * cscPhi
                       * (complex_T(0, -8) * (om0 * y * (c_t - z) * sinPhi2_2 * sinPhi_4)
                              * (complex_T(0, 1) * rho0 - z_sinPhi)
                          - om0 * sinPhi_4 * sinPhi
                              * (-float_t(2.0) * z2 * rho0
                                 - cspeed * cspeed
                                     * (k * tauG2 * x2 + float_t(2.0) * t * (t - complex_T(0, 1) * om0_tauG2) * rho0)
                                 + cspeed * (4 * t * z * rho0 - complex_T(0, 2) * (om0_tauG2 * z * rho0))
                                 - complex_T(0, 2) * (c_t - z) * (cspeed * (t - complex_T(0, 1) * om0_tauG2) - z)
                                     * z_sinPhi)
                          + y * sinPhi
                              * (complex_T(0, 4) * (om0 * y * (c_t - z) * sinPhi2_4)
                                 + om0 * (c_t - z) * (complex_T(0, 1) * c_t + c_Om0 * tauG2 - complex_T(0, 1) * z)
                                     * sinPhi_3
                                 - complex_T(0, 4) * sinPhi2_4
                                     * (cspeed * k * x2 - om0 * (y2 - float_T(4.0) * (c_t - z) * z) * sinPhi))
                              * sin2Phi
                          - complex_T(0, 4) * sinPhi2_4
                              * (complex_T(0, -4) * (om0 * y * (c_t - z) * rho0 * cosPhi * sinPhi_2)
                                 + complex_T(0, 2)
                                     * (om0 * (y2 + float_T(2.0) * z2) * rho0
                                        - cspeed * z * (complex_T(0, 1) * (k * x2) + float_T(2.0) * om0 * t * rho0))
                                     * sinPhi_3
                                 - float_T(2.0) * om0 * z * (y2 - float_T(2.0) * (c_t - z) * z) * sinPhi_4
                                 + om0 * y2 * (c_t - z) * sin2Phi * sin2Phi)))
                    / (cspeed * helpVar2 * helpVar1);

                const complex_T helpVar4 = c_Om0 * tauG * tauG
                    - complex_T(0, 8) * y * math::tan(float_T(PI / 2.0) - phiT) * (cscPhi * cscPhi * sinPhi2_4)
                    - complex_T(0, 2) * (z * tanPhi2_2);

                const complex_T result
                    = (math::exp(helpVar3) * tauG * secPhi2 * secPhi2
                       * (complex_T(0, 2) * c_t + c_Om0 * tauG2 - complex_T(0, 4) * z
                          + cspeed * (complex_T(0, 2) * t + om0_tauG2) * cosPhi + complex_T(0, 2) * (y * tanPhi2))
                       * math::sqrt(c_Om0 * rho0 / helpVar2))
                    / (float_T(2.0) * cspeed * math::pow(helpVar4, float_T(1.5)));

                return result.get_real() / UNIT_SPEED;
            }

            /** Calculate the Bz(r,t) field
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations)
             *             for calculating the field */
            HDINLINE BField::float_T BField::calcTWTSBz_Ex(const float3_64& pos, const float_64 time) const
            {
                using complex_T = pmacc::math::Complex<float_T>;

                /* If phi < 0 the formulas below are not directly applicable.
                 * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
                 * z-axis of the coordinate system in this function.
                 */
                const float_T phiReal = float_T(math::abs(phi));
                float_T sinPhiReal;
                float_T cosPhiReal;
                pmacc::math::sincos(phiReal, sinPhiReal, cosPhiReal);
                const float_T alphaTilt = math::atan2(float_T(1.0) - beta_0 * cosPhiReal, beta_0 * sinPhiReal);
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
                const float_T phiT = float_T(2.0) * alphaTilt;

                /* Angle between the laser pulse front and the y-axis.
                 * Not used, but remains in code for documentation purposes.
                 * const float_T eta = float_T(float_T(PI / 2)) - (phiReal - alphaTilt);
                 */

                const float_T cspeed = float_T(SI::SPEED_OF_LIGHT_SI / UNIT_SPEED);
                const float_T lambda0 = float_T(wavelength_SI / UNIT_LENGTH);
                const float_T om0 = float_T(2.0 * PI * cspeed / lambda0);
                /* factor 2  in tauG arises from definition convention in laser formula */
                const float_T tauG = float_T(pulselength_SI * 2.0 / UNIT_TIME);
                /* w0 is wx here --> w0 could be replaced by wx */
                const float_T w0 = float_T(w_x_SI / UNIT_LENGTH);
                const float_T rho0 = float_T(PI * w0 * w0 / lambda0);
                /* wy is width of TWTS pulse */
                const float_T wy = float_T(w_y_SI / UNIT_LENGTH);
                const float_T k = float_T(2.0 * PI / lambda0);

                /* In order to calculate in single-precision and in order to account for errors in
                 * the approximations far from the coordinate origin, we use the wavelength-periodicity and
                 * the known propagation direction for realizing the laser pulse using relative coordinates
                 * (i.e. from a finite coordinate range) only. All these quantities have to be calculated
                 * in double precision.
                 */
                float_64 sinPhiVal;
                float_64 cosPhiVal;
                pmacc::math::sincos(precisionCast<float_64>(phi), sinPhiVal, cosPhiVal);
                const float_64 tanAlpha = (float_64(1.0) - beta_0 * cosPhiVal) / (beta_0 * sinPhiVal);
                const float_64 tanFocalLine = math::tan(PI / float_64(2.0) - phi);
                const float_64 deltaT
                    = wavelength_SI / SI::SPEED_OF_LIGHT_SI * (float_64(1.0) + tanAlpha / tanFocalLine);
                const float_64 deltaY = wavelength_SI / tanFocalLine;
                const float_64 deltaZ = -wavelength_SI;
                const float_64 numberOfPeriods = math::floor(time / deltaT);
                const float_T timeMod = float_T(time - numberOfPeriods * deltaT);
                const float_T yMod = float_T(pos.y() + numberOfPeriods * deltaY);
                const float_T zMod = float_T(pos.z() + numberOfPeriods * deltaZ);

                /* Find out the envelope factor along the (long) TWTS pulse width (y-axis)
                 * according to a Tukey-window.
                 * This is only correct if the transition region size deltawy = wy * alpha / 2
                 * is very much larger than the characteristic size of diffraction in the simulation.
                 * It is useful to compare the actual TWTS propagation distance (within the simulation volume)
                 * with the corresponding "Rayleigh length" PI * deltawy * deltawy / lambda0.
                 */

                const float_T x = float_T(phiPositive * pos.x() / UNIT_LENGTH);
                const float_T y = float_T(phiPositive * yMod / UNIT_LENGTH);
                const float_T z = float_T(zMod / UNIT_LENGTH);
                const float_T t = float_T(timeMod / UNIT_TIME);

                /* This makes the pulse super-gaussian (to the power of 8) and removes the previous purely gaussian
                 * dependency. This is a hack), which can only work close to the center of the Rayleigh length rho0,
                 * because it does not include eventual phase-evolution, due to super-gaussian focusing instead of
                 * gaussian focusing.
                 */

                /* Calculating shortcuts for speeding up field calculation */
                float_T sinPhi;
                float_T cosPhi;
                pmacc::math::sincos(phiT, sinPhi, cosPhi);
                const float_t phiThalf = phiT / float_T(2.0);
                const float_T cscPhi = float_T(1.0) / sinPhi;
                const float_T secPhi2 = float_T(1.0) / math::cos(phiT / float_T(2.0));
                const float_T sinPhi2 = math::sin(phiThalf);
                const float_T tanPhi2 = math::tan(phiThalf);

                const float_T cscPhi_3 = cscPhi * cscPhi * cscPhi;

                const float_T sinPhi2_2 = sinPhi2 * sinPhi2;
                const float_T sinPhi2_4 = sinPhi2_2 * sinPhi2_2;
                const float_T tanPhi2_2 = tanPhi2 * tanPhi2;
                const float_T secPhi2_2 = secPhi2 * secPhi2;

                const float_T tanPI2_phi = math::tan(float_T(PI / 2.0) - phiT);

                const float_T tauG2 = tauG * tauG;
                const float_T om02 = om0 * om0;
                const float_T x2 = x * x;
                const float_T y2 = y * y;
                const float_T z2 = z * z;

                const float_T c_t = cspeed * t;
                const float_T c_Om0 = cspeed * om0;
                const float_T z_sinPhi = z * sinPhi;

                /* The "helpVar" variables decrease the nesting level of the evaluated expressions and
                 * thus help with formal code verification through manual code inspection.
                 */
                const complex_T helpVar1 = c_Om0 * tauG2 - complex_T(0, 1) * (y * cosPhi * secPhi2_2 * tanPhi2)
                    - complex_T(0, 2) * (z * tanPhi2_2);
                const complex_T helpVar2 = complex_T(0, 1) * (cspeed * rho0) - cspeed * y * cosPhi - cspeed * z_sinPhi;

                const complex_T helpVar3 = rho0 + complex_T(0, 1) * (y * cosPhi) + complex_T(0, 1) * z_sinPhi;
                const complex_T helpVar4 = complex_T(0, 1) * rho0 - y * cosPhi - z_sinPhi;
                const complex_T helpVar5 = -z - y * tanPI2_phi + complex_T(0, 1) * rho0 * cscPhi;
                const complex_T helpVar6
                    = -cspeed * z - cspeed * y * tanPI2_phi + complex_T(0, 1) * cspeed * rho0 * cscPhi;
                const complex_T helpVar7 = complex_T(0, 1) * (cspeed * rho0) - cspeed * y * cosPhi - cspeed * z_sinPhi;

                const complex_T helpVar8
                    = (om0 * y * rho0 * secPhi2_2 * secPhi2_2 / helpVar6
                       + (om0 * y * tanPI2_phi
                          * (c_Om0 * tauG2 + float_T(8.0) * (complex_T(0, 2) * y + rho0) * cscPhi_3 * sinPhi2_4))
                           / (cspeed * helpVar5)
                       + om02 * tauG2 * z_sinPhi / helpVar4 - float_T(2.0) * k * x2 / helpVar3
                       - om02 * tauG2 * rho0 / helpVar3
                       + complex_T(0, 1) * (om0 * y2 * cosPhi * cosPhi * secPhi2_2 * tanPhi2) / helpVar2
                       + complex_T(0, 4) * (om0 * y * z * tanPhi2_2) / helpVar2
                       - float_T(2.0) * om0 * z * rho0 * tanPhi2_2 / helpVar2
                       - complex_T(0, 2) * (om0 * z2 * sinPhi * tanPhi2_2) / helpVar2
                       - (om0
                          * math::pow(
                              float_T(2.0) * c_t - complex_T(0, 1) * c_Om0 * tauG2 - float_T(2.0) * z
                                  + float_T(8.0) * y * cscPhi_3 * sinPhi2_4 - float_T(2.0) * z * tanPhi2_2,
                              float_T(2.0)))
                           / (cspeed * helpVar1))
                    / float_T(4.0);

                const complex_T helpVar9 = c_Om0 * tauG2 - complex_T(0, 1) * (y * cosPhi * secPhi2_2 * tanPhi2)
                    - complex_T(0, 2) * (z * tanPhi2_2);

                const complex_T result = float_T(phiPositive)
                    * (complex_T(0, 2) * math::exp(helpVar8) * (tauG * tanPhi2 * (c_t - z + y * tanPhi2))
                       * math::sqrt(om0 * rho0 / helpVar7))
                    / math::pow(helpVar9, float_T(1.5));

                return result.get_real() / UNIT_SPEED;
            }

            /** Calculate the Bx(r,t) field
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations)
             *             for calculating the field */
            HDINLINE BField::float_T BField::calcTWTSBx(const float3_64& pos, const float_64 time) const
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
            /* CAUTION: In this branch, this fuction is not consistent with the other components.  */
            /* Thus TWTS polarization in for Ey is not available right now. */
            HDINLINE BField::float_T BField::calcTWTSBz_Ey(const float3_64& pos, const float_64 time) const
            {
                using complex_T = pmacc::math::Complex<float_T>;
                using complex_64 = pmacc::math::Complex<float_64>;

                /* If phi < 0 the formulas below are not directly applicable.
                 * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
                 * z-axis of the coordinate system in this function.
                 */
                const float_T phiReal = float_T(math::abs(phi));
                float_T sinPhiReal;
                float_T cosPhiReal;
                pmacc::math::sincos(phiReal, sinPhiReal, cosPhiReal);
                const float_T alphaTilt = math::atan2(float_T(1.0) - beta_0 * cosPhiReal, beta_0 * sinPhiReal);
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
                const float_T phiT = float_T(2.0) * alphaTilt;

                /* Angle between the laser pulse front and the y-axis.
                 * Not used, but remains in code for documentation purposes.
                 * const float_T eta = float_T(float_T(PI / 2)) - (phiReal - alphaTilt);
                 */

                const float_T cspeed = float_T(SI::SPEED_OF_LIGHT_SI / UNIT_SPEED);
                const float_T lambda0 = float_T(wavelength_SI / UNIT_LENGTH);
                const float_T om0 = float_T(2.0 * PI * cspeed / lambda0);
                /* factor 2  in tauG arises from definition convention in laser formula */
                const float_T tauG = float_T(pulselength_SI * 2.0 / UNIT_TIME);
                /* w0 is wx here --> w0 could be replaced by wx */
                const float_T w0 = float_T(w_x_SI / UNIT_LENGTH);
                const float_T rho0 = float_T(PI * w0 * w0 / lambda0);
                /* wy is width of TWTS pulse */
                const float_T wy = float_T(w_y_SI / UNIT_LENGTH);
                const float_T k = float_T(2.0 * PI / lambda0);
                /* If phi < 0 the entire pulse is rotated by 180 deg around the
                 * z-axis of the coordinate system without also changing
                 * the orientation of the resulting field vectors.
                 */
                const float_T x = float_T(phiPositive * pos.x() / UNIT_LENGTH);
                const float_T y = float_T(phiPositive * pos.y() / UNIT_LENGTH);
                const float_T z = float_T(pos.z() / UNIT_LENGTH);
                const float_T t = float_T(time / UNIT_TIME);

                /* Shortcuts for speeding up the field calculation. */
                float_T sinPhi;
                float_T cosPhi;
                pmacc::math::sincos(phiT, sinPhi, cosPhi);
                const float_t phiThalf = phiT / float_T(2.0);
                float_T sinPhi2;
                float_T cosPhi2;
                pmacc::math::sincos(phiThalf, sinPhi2, cosPhi2);

                const float_T tanPhi2 = math::tan(phiThalf);

                const float_T sinPhi2_2 = sinPhi2 * sinPhi2;
                const float_T sinPhi2_4 = sinPhi2_2 * sinPhi2_2;
                const float_T cosPhi2_2 = cosPhi2 * cosPhi2;
                const float_T tanPhi2_2 = tanPhi2 * tanPhi2;

                const float_T cspeed2 = cspeed * cspeed;
                const float_T tauG2 = tauG * tauG;
                const float_T wy2 = wy * wy;
                const float_T om02 = om0 * om0;

                const float_T x2 = x * x;
                const float_T y2 = y * y;
                const float_T z2 = z * z;
                const float_T t2 = t * t;

                const float_T c_t = cspeed * t;
                const float_T c_Om0 = cspeed * om0;
                const float_T om0_tauG2 = om0 * tauG2;
                const float_T om0_wy2 = om0 * wy2;
                const float_T om0_wy2_sinPhi = om0_wy2 * sinPhi;
                const float_T om02_wy2_sinPhi = om02 * wy2 * sinPhi;
                const float_T om0_wy2_roh0 = om0_wy2 * rho0;
                const float_T c2_om0_wy2_roh0 = cspeed2 * om0_wy2 * rho0;

                /* The "helpVar" variables decrease the nesting level of the evaluated expressions and
                 * thus help with formal code verification through manual code inspection.
                 */
                const complex_T helpVar1 = complex_T(0, -1) * (c_Om0 * tauG2) - y * cosPhi / cosPhi2_2 * tanPhi2
                    - float_T(2.0) * z * tanPhi2_2;
                const complex_T helpVar2 = complex_T(0, 1) * rho0 - y * cosPhi - z * sinPhi;


                const complex_T helpVar3
                    = (-cspeed2 * k * om0_tauG2 * wy2 * x2 - float_T(2.0) * t2 * c2_om0_wy2_roh0
                       + complex_T(0, 2) * (t * tauG2 * c2_om0_wy2_roh0)
                       - float_T(2.0) * cspeed2 * om0_tauG2 * y2 * rho0 + float_T(4.0) * c_t * z * om0_wy2_roh0
                       - complex_T(0, 2) * (cspeed * tauG2 * z * om0_wy2_roh0) - float_T(2.0) * z2 * om0_wy2_roh0
                       - complex_T(0, 8) * (om0_wy2 * y * (c_t - z) * z * sinPhi2_2)
                       + complex_T(0, 8) / sinPhi
                           * (float_T(2.0) * z2 * (c_t * om0_wy2 + complex_T(0, 1) * (cspeed * y2) - om0_wy2 * z)
                              + y
                                  * (cspeed * k * wy2 * x2 - complex_T(0, 2) * (c_t * om0_wy2_roh0)
                                     + float_T(2.0) * cspeed * y2 * rho0 + complex_T(0, 2) * z * om0_wy2_roh0)
                                  * tan(float_T(PI) / float_T(2.0) - phiT) / sinPhi)
                           * sinPhi2_4
                       - complex_T(0, 2) * (t2 * z * c2_om0_wy2_roh0)
                       - float_T(2.0) * cspeed2 * om02_wy2_sinPhi * t * tauG2 * wy2 * z
                       - complex_T(0, 2) * (cspeed2 * om0_tauG2 * y2 * z * sinPhi)
                       + complex_T(0, 4) * (c_t * z2 * om0_wy2_sinPhi)
                       + float_T(2.0) * cspeed * om02_wy2_sinPhi * tauG2 * wy2 * z2
                       - complex_T(0, 2) * (z2 * z * om0_wy2_sinPhi) - float_T(4.0) * c_t * y * om0_wy2_roh0 * tanPhi2
                       + float_T(4.0) * y * z * om0_wy2_roh0 * tanPhi2
                       + complex_T(0, 2) * y2 * (c_t * om0_wy2 + complex_T(0, 1) * cspeed * y2 - om0_wy2 * z)
                           * (cosPhi * cosPhi) / cosPhi2_2 * tanPhi2
                       + complex_T(0, 2) * (cspeed * k * wy2 * x2 * z * tanPhi2_2)
                       - float_T(2.0) * y2 * om0_wy2_roh0 * tanPhi2_2
                       + float_T(4.0) * c_t * z * om0_wy2_roh0 * tanPhi2_2
                       + complex_T(0, 4) * (cspeed * y2 * z * rho0 * tanPhi2_2)
                       - float_T(4.0) * z2 * om0_wy2_roh0 * tanPhi2_2
                       - complex_T(0, 2) * (y2 * z * om0_wy2_sinPhi * tanPhi2_2)
                       - float_T(2.0) * y * cosPhi
                           * (om0
                                  * (cspeed2
                                         * (complex_T(0, 1) * (t2 * wy2) + om0_wy2 * t * tauG2
                                            + complex_T(0, 1) * (tauG2 * y2))
                                     - cspeed * (complex_T(0, 2) * t + om0_tauG2) * wy2 * z
                                     + complex_T(0, 1) * (wy2 * z2))
                              + complex_T(0, 2) * (om0_wy2 * y * (c_t - z) * tanPhi2)
                              + complex_T(0, 1)
                                  * (complex_T(0, -4) * (cspeed * y2 * z)
                                     + om0_wy2 * (y2 - float_T(4.0) * (c_t - z) * z))
                                  * tanPhi2_2)
                       /* The "round-trip" conversion in the line below fixes a gross accuracy bug
                        * in floating-point arithmetics, when float_T is set to float_X.
                        */
                       )
                    * complex_T(float_64(1.0) / complex_64(float_T(2.0) * cspeed * wy2 * helpVar2 * helpVar1));

                const complex_T helpVar4
                    = (c_Om0
                       * (c_Om0 * tauG2
                          - complex_T(0, 8)
                              * (y * math::tan(float_T(PI) / float_T(2.0) - phiT) / sinPhi / sinPhi * sinPhi2_4)
                          - complex_T(0, 2) * (z * tanPhi2_2)))
                    / rho0;

                const complex_T result = float_T(phiPositive) * float_T(-1.0)
                    * (cspeed * math::exp(helpVar3) * k * tauG * x * math::pow(helpVar2, float_T(-1.5))
                       / math::sqrt(helpVar4));

                return result.get_real() / UNIT_SPEED;
            }

        } /* namespace twts */
    } /* namespace templates */
} /* namespace picongpu */
