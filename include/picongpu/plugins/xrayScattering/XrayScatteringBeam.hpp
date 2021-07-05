/* Copyright 2020-2021 Pawel Ordyna
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

#include "picongpu/param/xrayScattering.param"
#include "picongpu/plugins/externalBeam/CoordinateTransform.hpp"
#include "picongpu/plugins/externalBeam/ProbingBeam.hpp"
#include "picongpu/plugins/externalBeam/SecondaryRotation.hpp"
#include "picongpu/plugins/externalBeam/beamProfiles/profiles.hpp"
#include "picongpu/plugins/externalBeam/beamShapes/shapes.hpp"


namespace picongpu
{
    namespace plugins
    {
        namespace xrayScattering
        {
            namespace beam
            {
                // TODO: Move this back to the param file after fixing the coordinate
                // transform.
                constexpr float_X BEAM_OFFSET[2] = {0.0, 0.0};
                constexpr float_X BEAM_DELAY_SI = 0.0;
                using BeamProfile = externalBeam::beamProfiles::ConstProfile;
                using BeamShape = externalBeam::beamShapes::ConstShape;

                using BeamCoordinates = picongpu::plugins::externalBeam::CoordinateTransform<
                    ProbingSide,
                    picongpu::plugins::externalBeam::SecondaryRotation<RotationParam>>;
                using XrayScatteringBeam
                    = picongpu::plugins::externalBeam::ProbingBeam<BeamProfile, BeamShape, BeamCoordinates>;

            } // namespace beam
        } // namespace xrayScattering
    } // namespace plugins
} // namespace picongpu
