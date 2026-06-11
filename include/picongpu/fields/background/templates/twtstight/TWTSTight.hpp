/*
 * SPDX-FileCopyrightText: Alexander Debus, Axel Huebl, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file
 *
 * This background field implements a obliquely incident, cylindrically-focused, pulse-front tilted laser for some
 * incidence angle phi as used for [1].
 *
 * The TWTS implementation is based on the definition of eq. (7) in [1]. Additionally, techniques from [2] and [3]
 * are used to allow for strictly Maxwell-conform solutions for tight foci wx or small incident angles phi.
 *
 * Specifically, this TWTStight implementation assumes a special case, where the transverse extent (but not its height
 * wx or its pulse duration) of the TWTS-laser wy is assumed to be infinite. While this special case of the TWTS laser
 * applies to a large range of use cases, the resulting form uses different spatial and time coordinates
 * (timeMod and yMod), which allow long term numerical stability beyond 100000 timesteps at single precision,
 * as well as for mitigating errors of the approximations far from the coordinate origin.
 *
 * We exploit the wavelength-periodicity and the known propagation direction for realizing the laser pulse
 * using reduced coordinates (i.e. a finite coordinate range only). All these quantities have to be calculated
 * in double precision.
 *
 * float_64 const deltaT
 *     = wavelength_SI / sim.si.getSpeedOfLight() / (1.0 - beta_0 * math::cos(precisionCast<float_64>(phi)));
 * float_64 const deltaY = beta_0 * sim.si.getSpeedOfLight() * deltaT;
 * float_64 const numberOfPeriods = math::floor(time / deltaT);
 * auto const timeMod = float_T(time - numberOfPeriods * deltaT);
 * auto const yMod = float_T(pos.y() - numberOfPeriods * deltaY);
 *
 * Literature:
 * [1] Steiniger et al., "Optical free-electron lasers with Traveling-Wave Thomson-Scattering",
 *     Journal of Physics B: Atomic, Molecular and Optical Physics, Volume 47, Number 23 (2014),
 *     https://doi.org/10.1088/0953-4075/47/23/234011
 * [2] Mitri, F. G., "Cylindrical quasi-Gaussian beams", Opt. Lett., 38(22), pp. 4727-4730 (2013),
 *     https://doi.org/10.1364/OL.38.004727
 * [3] Hua, J. F., "High-order corrected fields of ultrashort, tightly focused laser pulses",
 *     Appl. Phys. Lett. 85, 3705-3707 (2004),
 *     https://doi.org/10.1063/1.1811384
 *
 */

#pragma once
#include "picongpu/defines.hpp"
#include "picongpu/fields/background/templates/twtstight/numComponents.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/types.hpp>

#include <array>
#include <tuple>

namespace picongpu::templates::twtstight
{
    using float_T = float_X;
    using float3_T = ::pmacc::math::Vector<float_T, 3u>;
    using complex_T = alpaka::Complex<float_T>;
    using complex_64 = alpaka::Complex<float_64>;
    /** To avoid underflows in computation, numsigmas controls where a zero cutoff is made.
     *  The fields thus are set to zero at a position (numSigmas * tauG * cspeed) ahead
     *  and behind the respective TWTS pulse envelope.
     *  Developer note: In case the float_T-type is set to float_64 instead of float_X,
     *  numSigma can be increased to numSigmas = 10 without running into numerical issues.
     */
    constexpr uint32_t numSigmas = 6;

    /** Provides the E- or B-field functors of the TWTSTight laser for the
     *  fieldBackground and incidentField approaches.
     *  @tparam T_Field Specializes for E- or B-Field
     *  @tparam T_dim Specializes for the simulation dimension
     */
    template<typename T_Field>
    struct TWTSTight
    {
    public:
        //! Helper method to define basic TWTS variables
        HDINLINE static std::array<float_T, 11u> defineBasicHelperVariables(
            float_64 const& phi,
            float_64 const& beta_0,
            float_64 const& wavelength_SI,
            float_64 const& pulselength_SI,
            float_64 const& w_x_SI);

        //! Helper method to define trigonometry shortcuts
        HDINLINE static std::array<float_T, 7u> defineTrigonometryShortcuts(
            std::array<float_T, 11u> const& basicTWTSHelperVariables,
            float_T const& polAngle);

        /** Center of simulation volume in number of cells */
        PMACC_ALIGN(halfSimSize, DataSpace<simDim> const);
        /** y-position of TWTS coordinate origin inside the simulation coordinates [meter]
            The other origin coordinates (x and z) default to globally centered values
            with respect to the simulation volume. */
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
        PMACC_ALIGN(phi, float_64 const);
        /** Takes value 1.0 for phi > 0 and -1.0 for phi < 0. */
        PMACC_ALIGN(phiPositive, float_X const);
        /** propagation speed of TWTS laser overlap
        normalized to the speed of light. [Default: beta0=1.0] */
        PMACC_ALIGN(beta_0, float_64 const);
        /** If auto_tdelay=FALSE, then a user defined delay is used. [second] */
        PMACC_ALIGN(tdelay_user_SI, float_64 const);
        /** Defines z-position offset of TWTS coordinate origin inside the simulation coordinates [meter]
            relative to the globally centered default value with respect to the simulation volume. */
        PMACC_ALIGN(focus_z_offset_SI, float_64 const);
        /** Make time step constant accessible to device. */
        PMACC_ALIGN(dt, float_64 const);
        /** Make length normalization constant accessible to device. */
        PMACC_ALIGN(unit_length, float_64 const);
        /** Should the TWTS laser delay be chosen automatically, such that
         *  the laser gradually enters the simulation volume? [Default: TRUE]
         */
        PMACC_ALIGN(auto_tdelay, bool const);
        /** TWTS laser time delay */
        PMACC_ALIGN(tdelay, float_64 const);
        /** Polarization of TWTS laser with respect to x-axis around propagation direction [rad, default = 0. *
         * (PI/180.)] */
        PMACC_ALIGN(polAngle, float_64 const);
        /* imaginary unit I */
        PMACC_ALIGN(I, complex_T const);
        PMACC_ALIGN(basicTWTSHelperVariables, std::array<float_T, 11u> const);
        PMACC_ALIGN(trigonometryShortcuts, std::array<float_T, 7u> const);

        /** Electric or magnetic field of the TWTS laser
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
         * @param focus_z_offset_SI the distance to the laser focus in z-direction [m]
         *  relative to the default simulation volume centered origin.
         * @param polAngle determines the TWTS laser polarization angle with respect to x-axis around
         * propagation direction [rad, default = 0. * (PI/180.)] Normal to laser pulse front tilt plane:
         * polAngle = 0.0 * (PI/180.) (linear polarization parallel to x-axis) Parallel to laser pulse front
         * tilt plane: polAngle = 90.0 * (PI/180.) (linear polarization parallel to yz-plane)
         */
        HINLINE
        TWTSTight(
            float_64 const focus_y_SI,
            float_64 const wavelength_SI,
            float_64 const pulselength_SI,
            float_64 const w_x_SI,
            float_64 const phi = 90. * (PI / 180.),
            float_64 const beta_0 = 1.0,
            float_64 const tdelay_user_SI = 0.0,
            bool const auto_tdelay = true,
            float_64 const focus_z_offset_SI = 0.0,
            float_64 const polAngle = 0. * (PI / 180.));

        /** Specify your background field E(r, t) or B(r, t) here
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

        /** Calculate the given component of E(r, t) or B(r, t)
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

        /** Calculate E(r, t) or B(r, t) for given position, time, and extra in-cell shifts
         *
         * @param cellIdx The total cell id counted from the start at t=0, note it is fractional
         * @param extraShifts The extra in-cell shifts to be added to calculate the position
         * @param currentStep The current time step for the field to be calculated at, note it is fractional
         */
        HDINLINE float3_X getValue(
            floatD_X const& cellIdx,
            pmacc::math::Vector<floatD_X, detail::numComponents> const& extraShifts,
            float_X const currentStep) const;

        //! Tests whether coordinates are within
        //! TWTSTight field envelope
        HDINLINE bool isOutsideTWTSEnvelope(std::array<float_T, 4u> const& minimalCoordinates) const;

        //! Helper method to define reduced TWTSTight coordinates
        HDINLINE std::array<float_T, 4u> defineMinimalCoordinates(float3_64 const& pos, float_64 const& time) const;

        //! Helper method to define common (complex-valued) variables
        //! for TWTSTight field calculation
        HDINLINE
        std::tuple<std::array<float_T, 8u>, std::array<complex_T, 6u>> defineCommonHelperVariables(
            std::array<float_T, 4u> const& minimalCoordinates) const;

        //! Defines the non-focused laser envelope (aka zero order term)
        //! for TWTSTight field calculation
        HDINLINE complex_T defineTWTSEnvelope(
            std::array<float_T, 4u> const& minimalCoordinates,
            std::tuple<std::array<float_T, 8u>, std::array<complex_T, 6u>> const& commonHelperVariables) const;

        /** Calculate the Ex(r,t) or Bx(r,t) field
         *
         * @param pos Spatial position of the target field
         * @param time Absolute time (SI, including all offsets and transformations)
         *  for calculating the field
         * @tparam T_Field Specializes for E- or B-Field
         * @return Ex- or Bx-field component of the TWTS field in SI units */
        HDINLINE float_T calcTWTSFieldX(float3_64 const& pos, float_64 const time) const;

        /** Calculate the Ey(r,t) or By(r,t) field
         *
         * @param pos Spatial position of the target field
         * @param time Absolute time (SI, including all offsets and transformations)
         *  for calculating the field
         * @tparam T_Field Specializes for E- or B-Field
         * @return Ey- or By-field component of the TWTS field in SI units */
        HDINLINE float_T calcTWTSFieldY(float3_64 const& pos, float_64 const time) const;

        /** Calculate the Ez(r,t) or Bz(r,t) field
         *
         * @param pos Spatial position of the target field
         * @param time Absolute time (SI, including all offsets and transformations)
         *  for calculating the field
         * @return Ez- or Bz-field component of the TWTS field in SI units */
        HDINLINE float_T calcTWTSFieldZ(float3_64 const& pos, float_64 const time) const;

        /** Calculate the E- or B-field vector of the TWTS laser in SI units.
         * @param cellIdx The total cell id counted from the start at timestep 0
         * @return E- or B-field vector of the TWTS field in SI units */
        HDINLINE float3_X getTWTSField_Normalized(
            pmacc::math::Vector<floatD_64, detail::numComponents> const& fieldPositions_SI,
            float_64 const time) const;
    };

} // namespace picongpu::templates::twtstight

#include "picongpu/fields/background/templates/twtstight/TWTSTight.tpp"
