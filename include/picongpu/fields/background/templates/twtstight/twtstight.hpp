/* Copyright 2014-2024 Alexander Debus
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
 * applies to a large range of use cases, the resulting form allows to use different spatial and time coordinates
 * (timeMod, yMod and zMod), which allow long term numerical stability beyond 100000 timesteps at single precision,
 * as well as for mitigating errors of the approximations far from the coordinate origin.
 *
 * We exploit the wavelength-periodicity and the known propagation direction for realizing the laser pulse
 * using relative coordinates (i.e. from a finite coordinate range) only. All these quantities have to be calculated
 * in double precision.
 *
 * float_64 const tanAlpha = (1.0 - beta_0 * math::cos(phi)) / (beta_0 * math::sin(phi));
 * float_64 const tanFocalLine = math::tan(PI / 2.0 - phi);
 * float_64 const deltaT = wavelength_SI / sim.si.getSpeedOfLight() * (1.0 + tanAlpha / tanFocalLine);
 * float_64 const deltaY = wavelength_SI * math::cos(phi) + wavelength_SI * math::sin(phi) * math::sin(phi) /
 * math::sin(phi); float_64 const numberOfPeriods = math::floor(time / deltaT); auto const timeMod = float_T(time -
 * numberOfPeriods * deltaT); auto const yMod = float_T(pos.y() - numberOfPeriods * deltaY);
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

#include "picongpu/fields/background/templates/twtstight/BField.hpp"
#include "picongpu/fields/background/templates/twtstight/EField.hpp"

namespace picongpu
{
    namespace templates
    {
        namespace twtstight
        {
            /** To avoid underflows in computation, numsigmas controls where a zero cutoff is made.
             *  The fields thus are set to zero at a position (numSigmas * tauG * cspeed) ahead
             *  and behind the respective TWTS pulse envelope.
             *  Developer note: In case the float_T-type is set to float_X instead of float_64,
             *  numSigma needs to be adjusted to numSigmas = 6 to avoid numerical issues.
             */
            constexpr uint32_t numSigmas = 10;
        } // namespace twtstight
    } // namespace templates
} // namespace picongpu

#include "picongpu/fields/background/templates/twtstight/twtstight.tpp"
