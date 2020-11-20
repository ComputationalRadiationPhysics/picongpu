/* Copyright 2014-2020 Alexander Debus, Axel Huebl
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

#include <pmacc/math/Vector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include "picongpu/fields/background/templates/gaussian/numComponents.hpp"

namespace picongpu
{
/* Load pre-defined background field */
namespace templates
{
/* Gaussian laser pulse */
namespace gaussian
{

class EField
{
public:
    using float_T = float_X;

    enum PolarizationType
    {
        /* The linear polarization of the Gaussian laser is defined
         * relative to the y-z-plane.
         *
         * Polarisation is normal to the reference plane.
         * Use Ex-fields (and corresponding B-fields) in Gaussian laser internal coordinate system.
         */
        LINEAR_X = 1u,
        /* Polarization lies within the reference plane.
         * Use Ey-fields (and corresponding B-fields) in Gaussian laser internal coordinate system.
         */
        LINEAR_YZ = 2u,
    };

    /* Center of simulation volume in number of cells */
    PMACC_ALIGN(halfSimSize,DataSpace<simDim>);
    /* y-position of Gaussian laser coordinate origin inside the simulation coordinates [meter]
       The other origin coordinates (x and z) default to globally centered values
       with respect to the simulation volume. */
    PMACC_ALIGN(focus_y_SI, const float_64);
    /* Laser wavelength [meter] */
    PMACC_ALIGN(wavelength_SI, const float_64);
    /* Gaussian laser pulse duration [second] */
    PMACC_ALIGN(pulselength_SI, const float_64);
    /* focal spot size of Gaussian laser pulse [meter] */
    PMACC_ALIGN(w0_SI, const float_64);
    /* interaction angle between Gaussian laser propagation vector and the y-axis [rad] */
    PMACC_ALIGN(phi, const float_X);
    /* If auto_tdelay=FALSE, then a user defined delay is used. [second] */
    PMACC_ALIGN(tdelay_user_SI, const float_64);
    /* Make time step constant accessible to device. */
    PMACC_ALIGN(dt, const float_64);
    /* Make length normalization constant accessible to device. */
    PMACC_ALIGN(unit_length, const float_64);
    /* Gaussian laser time delay */
    PMACC_ALIGN(tdelay,float_64);
    /* Should the Gaussian laser delay be chosen automatically, such that
     * the laser gradually enters the simulation volume? [Default: TRUE]
     */
    PMACC_ALIGN(auto_tdelay, const bool);
    /* Polarization of Gaussian laser */
    PMACC_ALIGN(pol, const PolarizationType);

    /** Electric field of the Gaussian laser
     *
     * \param focus_y_SI the distance to the laser focus in y-direction [m]
     * \param wavelength_SI central wavelength [m]
     * \param pulselength_SI sigma of std. gauss for intensity (E^2),
     *  pulselength_SI = FWHM_of_Intensity / 2.35482 [seconds (sigma)]
     * \param w0 beam waist: distance from the axis where the pulse electric field
     *  decreases to its 1/e^2-th part at the focus position of the laser [m]
     * \param phi interaction angle between Gaussian laser propagation vector and
     *  the y-axis [rad, default = 90.*(PI/180.)]
     * \param tdelay_user manual time delay if auto_tdelay is false
     * \param auto_tdelay calculate the time delay such that the Gaussian laser pulse is not
     *  inside the simulation volume at simulation start timestep = 0 [default = true]
     * \param pol determines the Gaussian laser polarization, which is either polarized
     *  along the x-axis or perpendicular within the YZ-plane [ default= LINEAR_X , LINEAR_YZ ]
     */
    HINLINE
    EField( const float_64 focus_y_SI,
            const float_64 wavelength_SI,
            const float_64 pulselength_SI,
            const float_64 w0_SI,
            const float_X phi               = 90.*(PI / 180.),
            const float_64 tdelay_user_SI   = 0.0,
            const bool auto_tdelay          = true,
            const PolarizationType pol      = LINEAR_X );

    /** Specify your background field E(r,t) here
     *
     * \param cellIdx The total cell id counted from the start at timestep 0.
     * \param currentStep The current time step
     * \return float3_X with field normalized to amplitude in range [-1.:1.]
     */
    HDINLINE float3_X
    operator()( const DataSpace<simDim>& cellIdx,
                const uint32_t currentStep ) const;

    /** Calculate the Ex(r,t) field for a laser pulse,
     *  which is linearly polarized along the x-axis.
     *
     * \param pos Spatial position of the target field
     * \param time Absolute time (SI, including all offsets and transformations)
     *  for calculating the field
     * \return Ex-field component of the non-rotated Gaussian laser field in SI units */
    HDINLINE float_T
    calcEx_Linear_X( const float3_64& pos, const float_64 time ) const;

    /** Calculate the Ey(r,t) field for a laser pulse,
     *  which is linearly polarized along the x-axis.
     *
     * \param pos Spatial position of the target field
     * \param time Absolute time (SI, including all offsets and transformations)
     *  for calculating the field
     * \return Ex-field component of the non-rotated Gaussian laser field in SI units */
    HDINLINE float_T
    calcEy_Linear_X( const float3_64& pos, const float_64 time ) const;

    /** Calculate the Ez(r,t) field for a laser pulse,
     *  which is linearly polarized along the x-axis.
     *
     * \param pos Spatial position of the target field
     * \param time Absolute time (SI, including all offsets and transformations)
     *  for calculating the field
     * \return Ex-field component of the non-rotated Gaussian laser field in SI units */
    HDINLINE float_T
    calcEz_Linear_X( const float3_64& pos, const float_64 time ) const;

    /** Calculate the Ex(r,t) field for a laser pulse,
     *  which is linearly polarized within the YZ-plane,
     *  and perpendicular to the x-axis.
     *
     * \param pos Spatial position of the target field
     * \param time Absolute time (SI, including all offsets and transformations)
     *  for calculating the field
     * \return Ex-field component of the non-rotated Gaussian laser field in SI units */
    HDINLINE float_T
    calcEx_Linear_YZ( const float3_64& pos, const float_64 time ) const;

    /** Calculate the Ey(r,t) field for a laser pulse,
     *  which is linearly polarized within the YZ-plane,
     *  and perpendicular to the x-axis.
     *
     * \param pos Spatial position of the target field
     * \param time Absolute time (SI, including all offsets and transformations)
     *  for calculating the field
     * \return Ex-field component of the non-rotated Gaussian laser field in SI units */
    HDINLINE float_T
    calcEy_Linear_YZ( const float3_64& pos, const float_64 time ) const;

    /** Calculate the Ez(r,t) field for a laser pulse,
     *  which is linearly polarized within the YZ-plane,
     *  and perpendicular to the x-axis.
     *
     * \param pos Spatial position of the target field
     * \param time Absolute time (SI, including all offsets and transformations)
     *  for calculating the field
     * \return Ex-field component of the non-rotated Gaussian laser field in SI units */
    HDINLINE float_T
    calcEz_Linear_YZ( const float3_64& pos, const float_64 time ) const;

    /** Calculate the E-field vector of the Gaussian laser in SI units
     *  with linear polarization along the x-axis.
     * \param cellIdx The total cell id counted from the start at timestep 0
     * \return Efield vector of the rotated Gaussian laser field in SI units */
    HDINLINE float3_X
    getEfield_Normalized_Linear_X(
            const pmacc::math::Vector<floatD_64,detail::numComponents>& eFieldPositions_SI,
            const float_64 time) const;

    /** Calculate the E-field vector of the Gaussian laser in SI units
     *  with linear polarization within the YZ-plane.
     * \param cellIdx The total cell id counted from the start at timestep 0
     * \return Efield vector of the rotated Gaussian laser field in SI units */
    HDINLINE float3_X
    getEfield_Normalized_Linear_YZ(
            const pmacc::math::Vector<floatD_64,detail::numComponents>& eFieldPositions_SI,
            const float_64 time) const;

};

} /* namespace gaussian */
} /* namespace templates */
} /* namespace picongpu */
