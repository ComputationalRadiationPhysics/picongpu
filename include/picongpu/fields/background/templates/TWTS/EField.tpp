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
#include "picongpu/fields/background/templates/TWTS/EField.hpp"
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
    /* Load pre-defined background field */
    namespace templates
    {
        /* Traveling-wave Thomson scattering laser pulse */
        namespace twts
        {
            HINLINE
            EField::EField(
                const float_64 focus_y_SI,
                const float_64 wavelength_SI,
                const float_64 pulselength_SI,
                const float_64 w_x_SI,
                const float_64 w_y_SI,
                const float_X phi,
                const float_X beta_0,
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
                , phiPositive(float_X(1.0))
            {
                /* Note: Enviroment-objects cannot be instantiated on CUDA GPU device. Since this is done
                         on host (see fieldBackground.param), this is no problem.
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
                if(phi < float_X(0.0))
                    phiPositive = float_X(-1.0);
            }

            template<>
            HDINLINE float3_X EField::getTWTSEfield_Normalized<DIM3>(
                const pmacc::math::Vector<floatD_64, detail::numComponents>& eFieldPositions_SI,
                const float_64 time) const
            {
                float3_64 pos(float3_64::create(0.0));
                for(uint32_t i = 0; i < simDim; ++i)
                    pos[i] = eFieldPositions_SI[0][i];
                return float3_X(float_X(calcTWTSEx(pos, time)), float_X(0.), float_X(0.));
            }

            template<>
            HDINLINE float3_X EField::getTWTSEfield_Normalized_Ey<DIM3>(
                const pmacc::math::Vector<floatD_64, detail::numComponents>& eFieldPositions_SI,
                const float_64 time) const
            {
                typedef pmacc::math::Vector<float3_64, detail::numComponents> PosVecVec;
                PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));

                for(uint32_t k = 0; k < detail::numComponents; ++k)
                {
                    for(uint32_t i = 0; i < simDim; ++i)
                        pos[k][i] = eFieldPositions_SI[k][i];
                }

                /* Calculate Ey-component with the intra-cell offset of a Ey-field */
                const float_64 Ey_Ey = calcTWTSEy(pos[1], time);
                /* Calculate Ey-component with the intra-cell offset of a Ez-field */
                const float_64 Ey_Ez = calcTWTSEy(pos[2], time);

                /* Since we rotated all position vectors before calling calcTWTSEy,
                 * we need to back-rotate the resulting E-field vector.
                 *
                 * RotationMatrix[-(PI/2+phi)].(Ey,Ez) for rotating back the field-vectors.
                 */
                const float_64 Ey_rot = -math::sin(+phi) * Ey_Ey;
                const float_64 Ez_rot = -math::cos(+phi) * Ey_Ez;

                /* Finally, the E-field normalized to the peak amplitude. */
                return float3_X(float_X(0.0), float_X(Ey_rot), float_X(Ez_rot));
            }

            template<>
            HDINLINE float3_X EField::getTWTSEfield_Normalized<DIM2>(
                const pmacc::math::Vector<floatD_64, detail::numComponents>& eFieldPositions_SI,
                const float_64 time) const
            {
                /* Ex->Ez, so also the grid cell offset for Ez has to be used. */
                float3_64 pos(float3_64::create(0.0));
                /* 2D (y,z) vectors are mapped on 3D (x,y,z) vectors. */
                for(uint32_t i = 0; i < DIM2; ++i)
                    pos[i + 1] = eFieldPositions_SI[2][i];
                return float3_X(float_X(0.), float_X(0.), float_X(calcTWTSEx(pos, time)));
            }

            template<>
            HDINLINE float3_X EField::getTWTSEfield_Normalized_Ey<DIM2>(
                const pmacc::math::Vector<floatD_64, detail::numComponents>& eFieldPositions_SI,
                const float_64 time) const
            {
                typedef pmacc::math::Vector<float3_64, detail::numComponents> PosVecVec;
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
                const float_64 Ey_Ey = calcTWTSEy(pos[1], time);
                /* Calculate Ey-component with the intra-cell offset of a Ex-field */
                const float_64 Ey_Ex = calcTWTSEy(pos[0], time);

                /* Since we rotated all position vectors before calling calcTWTSEy,
                 * we need to back-rotate the resulting E-field vector.
                 *
                 * RotationMatrix[-(PI / 2+phi)].(Ey,Ex) for rotating back the field-vectors.
                 */
                const float_64 Ey_rot = -math::sin(+phi) * Ey_Ey;
                const float_64 Ex_rot = -math::cos(+phi) * Ey_Ex;

                /* Finally, the E-field normalized to the peak amplitude. */
                return float3_X(float_X(Ex_rot), float_X(Ey_rot), float_X(0.0));
            }

            HDINLINE float3_X EField::operator()(const DataSpace<simDim>& cellIdx, const uint32_t currentStep) const
            {
                const float_64 time_SI = float_64(currentStep) * dt - tdelay;
                const traits::FieldPosition<fields::CellType, FieldE> fieldPosE;

                const pmacc::math::Vector<floatD_64, detail::numComponents> eFieldPositions_SI
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
            HDINLINE EField::float_T EField::calcTWTSEx(const float3_64& pos, const float_64 time) const
            {
                using complex_T = pmacc::math::Complex<float_T>;
                using complex_64 = pmacc::math::Complex<float_64>;
                /* Unit of speed */
                const float_64 UNIT_SPEED = SI::SPEED_OF_LIGHT_SI;
                /* Unit of time */
                const float_64 UNIT_TIME = SI::DELTA_T_SI;
                /* Unit of length */
                const float_64 UNIT_LENGTH = UNIT_TIME * UNIT_SPEED;

                /* Propagation speed of overlap normalized to the speed of light [Default: beta0=1.0] */
                const float_T beta0 = float_T(beta_0);
                /* If phi < 0 the formulas below are not directly applicable.
                 * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
                 * z-axis of the coordinate system in this function.
                 */
                const float_T phiReal = float_T(math::abs(phi));
                const float_T alphaTilt = math::atan2(
                    float_T(1.0) - float_T(beta_0) * math::cos(phiReal),
                    float_T(beta_0) * math::sin(phiReal));
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
                const float_T phiT = float_T(2.0) * alphaTilt;

                /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for
                 * documentation purposes.
                 * const float_T eta = (PI / 2) - (phiReal - alphaTilt);
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
                const float_64 tanAlpha = (float_64(1.0) - beta_0 * math::cos(phi)) / (beta_0 * math::sin(phi));
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
                // float_T envelopeWy;
                // const float_T alpha = float_T(0.05);
                ////const float_T currentEnvelopePos = float_T(time / UNIT_TIME * cspeed); //Physical correct scenario
                // const float_T currentEnvelopePos = float_T(time / UNIT_TIME * cspeed); //Artificially eliminate
                // ponderomotive force from longitudinal envelope if ( ( -wy / float_T(2.0) <= currentEnvelopePos ) && (
                // currentEnvelopePos < ( alpha - float_T(1.0) ) * wy / float_T(2.0) ) )
                //{
                //    envelopeWy = float_T(0.5) * ( float_T(1.0) + math::cos( PI * ( ( float_T(2.0) *
                //    currentEnvelopePos + wy ) / ( alpha * wy ) - float_T(1.0) ) ) );
                //}
                // else if ( ( ( alpha - float_T(1.0) ) * wy / float_T(2.0) <= currentEnvelopePos ) &&  (
                // currentEnvelopePos <= ( float_T(1.0) - alpha ) * wy / float_T(2.0) ) )
                //{
                //    envelopeWy = float_T(1.0);
                //}
                // else if ( ( ( float_T(1.0) - alpha ) * wy / float_T(2.0) < currentEnvelopePos ) && (
                // currentEnvelopePos <= wy / float_T(2.0) ) )
                //{
                //    envelopeWy = float_T(0.5) * ( float_T(1.0) + math::cos( PI * ( ( float_T(2.0) *
                //    currentEnvelopePos + wy ) / ( alpha * wy ) - float_T(2.0) / alpha + float_T(1.0) ) ) );
                //}
                // else
                //{
                //    envelopeWy = float_T(0.0);
                //}

                const float_T x = float_T(phiPositive * pos.x() / UNIT_LENGTH);
                const float_T y = float_T(phiPositive * yMod / UNIT_LENGTH);
                const float_T z = float_T(zMod / UNIT_LENGTH);
                const float_T t = float_T(timeMod / UNIT_TIME);

                /* This makes the pulse super-gaussian (to the power of 8) and removes the previous purely gaussian
                 * dependency. This is a hack), which can only work close to the center of the Rayleigh length rho0,
                 * because it does not include eventual phase-evolution, due to super-gaussian focusing instead of
                 * gaussian focusing.
                 */
                // const float_T s = y * math::cos(phiT) + z * math::sin(phiT); // Formally, this probably includes a
                // sign-error, but this is OK, because s is later used only as s*s. const float_T wx2_s = w0 * w0 * (
                // float_T(1.0) + s*s / ( rho0 * rho0 ) ); const float_T envelopeWx = math::exp( +(x*x/wx2_s) -
                // math::pow( (x*x/wx2_s) , 4 ) );

                /* Calculating shortcuts for speeding up field calculation */
                const float_T sinPhi = math::sin(phiT);
                const float_T cosPhi = math::cos(phiT);
                const float_T cscPhi = float_T(1.0) / math::sin(phiT);
                const float_T sinPhi2 = math::sin(phiT / float_T(2.0));
                const float_T sin2Phi = math::sin(phiT * float_T(2.0));
                const float_T tanPhi2 = math::tan(phiT / float_T(2.0));

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

                /* The "helpVar" variables decrease the nesting level of the evaluated expressions and
                 * thus help with formal code verification through manual code inspection.
                 */
                const complex_T helpVar1 = cspeed * om0 * tauG2 * sinPhi_4
                    - complex_T(0, 8) * sinPhi2_4 * sinPhi * (y * cosPhi + z * sinPhi);

                const complex_T helpVar2 = complex_T(0, 1) * rho0 - y * cosPhi - z * sinPhi;

                const complex_T helpVar3 = complex_T(0, float_T(-0.5)) * cscPhi
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

                const complex_T helpVar4 = cspeed * om0 * tauG2
                    - complex_T(0, 8) * y * math::tan(float_T(PI / 2.0) - phiT) * cscPhi * cscPhi * sinPhi2_4
                    - complex_T(0, 2) * z * tanPhi2_2;

                const complex_T result
                    = (math::exp(helpVar3) * tauG * math::sqrt(cspeed * om0 * rho0 / helpVar2)) / math::sqrt(helpVar4);

                // return envelopeWx * envelopeWy * result.get_real();
                return result.get_real();
            }

            /** Calculate the Ey(r,t) field here
             *
             * @param pos Spatial position of the target field.
             * @param time Absolute time (SI, including all offsets and transformations) for calculating
             *             the field */
            HDINLINE EField::float_T EField::calcTWTSEy(const float3_64& pos, const float_64 time) const
            {
                /* The field function of Ey (polarization in pulse-front-tilt plane)
                 * is by definition identical to Ex (polarization normal to pulse-front-tilt plane)
                 */
                return calcTWTSEx(pos, time);
            }

        } /* namespace twts */
    } /* namespace templates */
} /* namespace picongpu */
