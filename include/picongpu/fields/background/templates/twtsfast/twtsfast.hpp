/* Copyright 2014-2021 Alexander Debus
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

/** @file
 *
 * This background field implements a obliquely incident, cylindrically-focused, pulse-front tilted laser for some
 * incidence angle phi as used for [1].
 *
 * The TWTS implementation generally follows the definition of eq. (7) in [1]. In deriving the magnetic field
 * components, a slowly-varying wave approximation was assumed, by neglegting the spatial derivatives of the
 * 2nd omega-order TWTS-phase-terms for the B-field-component transverse to direction of propagation, and additionally
 * neglect the 1st-order TWTS-phase-terms for the B-field-component longitudinal to the direction of propagation.
 *
 * Specifically, this TWTSfast approximation assumes a special case, where the transverse extent (but not its height wx
 * or its pulse duration) of the TWTS-laser wy is assumed to be infinite. While this special case of the TWTS laser
 * applies to a large range of use cases, the resulting form allows to use different spatial and time coordinates
 * (timeMod, yMod and zMod), which allow long term numerical stability beyond 100000 timesteps at single precision,
 * as well as for mitigating errors of the approximations far from the coordinate origin.
 *
 * We exploit the wavelength-periodicity and the known propagation direction for realizing the laser pulse
 * using relative coordinates (i.e. from a finite coordinate range) only. All these quantities have to be calculated
 * in double precision.
 *
 * float_64 const tanAlpha = (float_64(1.0) - beta_0 * math::cos(phi)) / (beta_0 * math::sin(phi));
 * float_64 const tanFocalLine = math::tan(PI / float_64(2.0) - phi);
 * float_64 const deltaT = wavelength_SI / SI::SPEED_OF_LIGHT_SI * (float_64(1.0) + tanAlpha / tanFocalLine);
 * float_64 const deltaY = wavelength_SI / tanFocalLine;
 * float_64 const deltaZ = -wavelength_SI;
 * float_64 const numberOfPeriods = math::floor(time / deltaT);
 * float_T const timeMod = float_T(time - numberOfPeriods * deltaT);
 * float_T const yMod = float_T(pos.y() + numberOfPeriods * deltaY);
 * float_T const zMod = float_T(pos.z() + numberOfPeriods * deltaZ);
 *
 * Literature:
 * [1] Steiniger et al., "Optical free-electron lasers with Traveling-Wave Thomson-Scattering",
 *     Journal of Physics B: Atomic, Molecular and Optical Physics, Volume 47, Number 23 (2014),
 *     https://doi.org/10.1088/0953-4075/47/23/234011
 */

#pragma once

#include "picongpu/fields/background/templates/twtsfast/EField.hpp"
#include "picongpu/fields/background/templates/twtsfast/BField.hpp"
