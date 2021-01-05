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
#include "picongpu/plugins/xrayScattering/beam/CoordinateTransform.hpp"

namespace picongpu
{
    namespace plugins
    {
        namespace xrayScattering
        {
            namespace beam
            {
                /** Defines the probing beam characteristic.
                 *
                 * @tparam T_BeamProfile Beam transverse profile.
                 * @tparam T_BeamShape Beam temporal shape.
                 * @tparam T_CoordinateTransform Coordinate transform from the pic
                 *      coordinate system to the beam coordinate system.
                 */
                template<typename T_BeamProfile, typename T_BeamShape, typename T_CoordinateTransform>
                struct ProbingBeam
                {
                    using BeamProfile = T_BeamProfile;
                    using BeamShape = T_BeamShape;
                    PMACC_ALIGN(coordinateTransform, T_CoordinateTransform);

                    HINLINE ProbingBeam() : coordinateTransform(){};

                    /** Calculates the probing amplitude at a given position.
                     * @param position_b Position in the beam comoving coordinate system
                     *      (x, y, z__at_t_0 - c*t).
                     * @returns Probing wave amplitude scaling at position_b.
                     */
                    HDINLINE float_X operator()(float3_X const& position_b)
                    {
                        float_X profileFactor = BeamProfile::getFactor(position_b[0], position_b[1]);

                        float_X beamTime = position_b[2] / SPEED_OF_LIGHT;
                        float_X shapeFactor = BeamShape::getFactor(beamTime);

                        return profileFactor * shapeFactor;
                    }
                };
            } // namespace beam
        } // namespace xrayScattering
    } // namespace plugins
} // namespace picongpu
