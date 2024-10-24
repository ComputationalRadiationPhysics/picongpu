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
#include "picongpu/fields/background/templates/twtstight/numComponents.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
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
            class BField
            {
            public:
                using float_T = float_64;

                /** Center of simulation volume in number of cells */
                PMACC_ALIGN(halfSimSize, DataSpace<simDim>);
                /** y-position of TWTS coordinate origin inside the simulation coordinates [meter]
                 *  The other origin coordinates (x and z) default to globally centered values
                 *  with respect to the simulation volume.
                 */
                PMACC_ALIGN(focus_y_SI, float_64 const);
                /** Laser wavelength [meter] */
                PMACC_ALIGN(wavelength_SI, float_64 const);
                /** TWTS laser pulse duration [second] */
                PMACC_ALIGN(pulselength_SI, float_64 const);
                /** line focus height of TWTS pulse [meter] */
                PMACC_ALIGN(w_x_SI, float_64 const);
                /** TWTS interaction angle
                 *  Enclosed by the laser propagation direction and the y-axis.
                 *  For a positive value of the interaction angle, the laser propagation direction
                 *  points along the y-axis and against the z-axis.
                 *  That is, for phi = 90 degree the laser propagates in the -z direction.
                 * [rad]
                 */
                PMACC_ALIGN(phi, float_X const);
                /** Takes value 1.0 for phi > 0 and -1.0 for phi < 0. */
                PMACC_ALIGN(phiPositive, float_X);
                /** propagation speed of TWTS laser overlap
                    normalized to the speed of light. [Default: beta0 = 1.0] */
                PMACC_ALIGN(beta_0, float_X const);
                /** If auto_tdelay=FALSE, then a user defined delay is used. [second] */
                PMACC_ALIGN(tdelay_user_SI, float_64 const);
                /** Make time step constant accessible to device. */
                PMACC_ALIGN(dt, float_64 const);
                /** Make length normalization constant accessible to device. */
                PMACC_ALIGN(unit_length, float_64 const);
                /** TWTS laser time delay */
                PMACC_ALIGN(tdelay, float_64);
                /** Should the TWTS laser time delay be chosen automatically, such that
                 * the laser gradually enters the simulation volume? [Default: TRUE]
                 */
                PMACC_ALIGN(auto_tdelay, bool const);
                /** Polarization angle of TWTS laser with respect to x-axis rotated around propagation direction. */
                PMACC_ALIGN(polAngle, float_X const);

                /** Magnetic field of the TWTS laser
                 *
                 * @param focus_y_SI the distance to the laser focus in y-direction [m]
                 * @param wavelength_SI central wavelength [m]
                 * @param pulselength_SI sigma of std. gauss for intensity (E^2),
                 *  pulselength_SI = FWHM_of_Intensity / 2.35482 [seconds (sigma)]
                 * @param w_x beam waist: distance from the axis where the pulse electric field
                 *  decreases to its 1/e^2-th part at the focus position of the laser [m]
                 * @param phi interaction angle between TWTS laser propagation vector and
                 *  the y-axis [rad, default = 90. * (PI/180.)]
                 * @param beta_0 propagation speed of overlap normalized to
                 *  the speed of light [c, default = 1.0]
                 * @param tdelay_user manual time delay if auto_tdelay is false
                 * @param auto_tdelay calculate the time delay such that the TWTS pulse is not
                 *  inside the simulation volume at simulation start timestep = 0 [default = true]
                 * @param polAngle determines the TWTS laser polarization angle with respect to x-axis around
                 * propagation direction [rad, default = 0. * (PI/180.)] Normal to laser pulse front tilt plane:
                 * polAngle = 0.0 * (PI/180.) (linear polarization parallel to x-axis) Parallel to laser pulse front
                 * tilt plane: polAngle = 90.0 * (PI/180.) (linear polarization parallel to yz-plane)
                 */
                HINLINE
                BField(
                    float_64 const focus_y_SI,
                    float_64 const wavelength_SI,
                    float_64 const pulselength_SI,
                    float_64 const w_x_SI,
                    float_X const phi = 90. * (PI / 180.),
                    float_X const beta_0 = 1.0,
                    float_64 const tdelay_user_SI = 0.0,
                    bool const auto_tdelay = true,
                    float_X const polAngle = 0. * (PI / 180.));


                /** Specify your background field B(r,t) here
                 *
                 * @param cellIdx The total cell id counted from the start at t=0, note it can be fractional
                 * @param currentStep The current time step for the field to be calculated at, note it can be
                 * fractional
                 * @return float3_X with field normalized to amplitude in range [-1.:1.]
                 *
                 * @{
                 */

                //! Integer index version, adds in-cell shifts according to the grid used; t = currentStep * dt
                //! This interface is used by the fieldBackground approach for implementing fields.
                HDINLINE float3_X operator()(DataSpace<simDim> const& cellIdx, uint32_t const currentStep) const;

                //! Floating-point index version, uses fractional cell index as provided; t = currentStep * dt
                //! This interface is used by the incidentField approach for implementing fields.
                HDINLINE float3_X operator()(floatD_X const& cellIdx, float_X const currentStep) const;

                /** @} */

                /** Calculate the given component of B(r, t)
                 *
                 * Result is same as for the fractional version of operator()(cellIdx, currentStep)[T_component].
                 * This version exists for optimizing usage in incident field where single components are needed.
                 *
                 * @tparam T_component field component, 0 = x, 1 = y, 2 = z
                 *
                 * @param cellIdx The total fractional cell id counted from the start at t=0
                 * @param currentStep The current time step for the field to be calculated at
                 * @return float_X with field component normalized to amplitude in range [-1.:1.]
                 */
                template<uint32_t T_component>
                HDINLINE float_X getComponent(floatD_X const& cellIdx, float_X const currentStep) const;

                /** Calculate B(r, t) for given position, time, and extra in-cell shifts
                 *
                 * @param cellIdx The total cell id counted from the start at t=0, note it is fractional
                 * @param extraShifts The extra in-cell shifts to be added to calculate the position
                 * @param currentStep The current time step for the field to be calculated at, note it is fractional
                 */
                HDINLINE float3_X getValue(
                    floatD_X const& cellIdx,
                    pmacc::math::Vector<floatD_X, detail::numComponents> const& extraShifts,
                    float_X const currentStep) const;

                /** Calculate the By(r,t) field
                 *
                 * @param pos Spatial position of the target field.
                 * @param time Absolute time (SI, including all offsets and transformations)
                 *  for calculating the field
                 * @return By-field component of the TWTS field in SI units */
                HDINLINE float_T calcTWTSBy(float3_64 const& pos, float_64 const time) const;

                /** Calculate the Bz(r,t) field
                 *
                 * @param pos Spatial position of the target field.
                 * @param time Absolute time (SI, including all offsets and transformations)
                 *  for calculating the field
                 * @return Bz-field component of the TWTS field in SI units */
                HDINLINE float_T calcTWTSBz(float3_64 const& pos, float_64 const time) const;

                /** Calculate the Bx(r,t) field
                 *
                 * @param pos Spatial position of the target field.
                 * @param time Absolute time (SI, including all offsets and transformations)
                 *  for calculating the field
                 * @return Bx-field component of the TWTS field in SI units */
                HDINLINE float_T calcTWTSBx(float3_64 const& pos, float_64 const time) const;

                /** Calculate the B-field vector of the TWTS laser in SI units.
                 * @tparam T_dim Specializes for the simulation dimension
                 * @param cellIdx The total cell id counted from the start at timestep 0
                 * @return B-field vector of the TWTS field in SI units */
                template<unsigned T_dim>
                HDINLINE float3_X getTWTSBfield_Normalized(
                    pmacc::math::Vector<floatD_64, detail::numComponents> const& bFieldPositions_SI,
                    float_64 const time) const;
            };

        } /* namespace twtstight */
    } /* namespace templates */
} /* namespace picongpu */
