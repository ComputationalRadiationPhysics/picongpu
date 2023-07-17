/* Copyright 2020-2023 Pawel Ordyna
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

// #include "picongpu/particles/externalBeam/beam/CoordinateTransform.hpp"
#include "picongpu/particles/externalBeam/beam/ProbingBeam.def"
#include "picongpu/particles/externalBeam/beam/SqrtWrapper.hpp"
#include "picongpu/particles/externalBeam/beam/beamProfiles/profiles.hpp"
#include "picongpu/particles/externalBeam/beam/beamShapes/shapes.hpp"


namespace picongpu::particles::externalBeam::beam
{
    //! Get the global domain size as a 3D float vector.
    HINLINE float3_X getDomainSize()
    {
        if constexpr(simDim == DIM2)
        {
            auto globalDomainSize{precisionCast<float_X>(Environment<DIM2>::get().SubGrid().getGlobalDomain().size)};
            return float3_X{globalDomainSize[0], globalDomainSize[1], 1.0_X};
        }
        else
        {
            return precisionCast<float_X>(Environment<DIM3>::get().SubGrid().getGlobalDomain().size);
        }
    };

    namespace defaults
    {
        struct DefaultOffsetParam
        {
            static constexpr float_X beamOffsetX_SI{0.0_X};
            static constexpr float_X beamOffsetY_SI{0.0_X};
            static constexpr float_X beamDelay_SI{0.0_X};
        };
    } // namespace defaults

    /** Defines a coordinate transform from the PIC system into the beam system.
     *

     */

    /** Defines the probing beam characteristic.
     *
     * @tparam T_BeamProfile Beam transverse profile.
     * @tparam T_BeamShape Beam temporal shape.
     * @tparam T_Side Side from which the probing beam is shot at the simulation box.
     * @tparam T_OffsetParam Param class defining spatial and temporal offsets (default is all set to 0).
     *  Example:
     *  @code{.cpp}
     *  struct DefaultOffsetParam
     *  {
     *      static constexpr float_X beamOffsetX_SI{0.0_X};
     *      static constexpr float_X beamOffsetY_SI{0.0_X};
     *      static constexpr float_X beamDelay_SI{0.0_X};
     *  };
     *  @endcode
     */
    template<typename T_BeamProfile, typename T_BeamShape, typename T_Side, typename T_OffsetParam>
    struct ProbingBeam
    {
        using BeamProfile = T_BeamProfile;
        using BeamShape = T_BeamShape;
        using Side = T_Side;
        using OffsetParam = T_OffsetParam;

        HINLINE ProbingBeam()
        {
            // Find the coordinate system translation:
            // Starting in the beam coordinate system.
            // Transverse(to the beam propagation direction) offset from the
            // initial position (the middle of the simulation box side).
            const float2_X offsetTrans_b = OffsetParam().beamOffset_SI / UNIT_LENGTH;
            // Offset along the propagation direction, defined by the beam
            // delay.
            const float_X beamDelay = OffsetParam::beamDelay_SI / UNIT_TIME;
            const float_X offsetParallel_b = beamDelay * SPEED_OF_LIGHT;
            // Complete offset from the initial position.
            const float3_X offsetFromMiddlePoint_b{offsetTrans_b[0], offsetTrans_b[1], -1.0_X * offsetParallel_b};

            // Move to the PIC coordinate system.
            const float3_X offsetFromMiddlePoint_s = Side::rotateBeamToSim(offsetFromMiddlePoint_b);

            // Find the initial position in the PIC coordinate system.
            float3_X toMiddlePoint_s = cellSize * getDomainSize();
            for(uint32_t ii = 0; ii < 3; ii++)
            {
                toMiddlePoint_s[ii] *= Side::beamStartPosition[ii];
            }
            // Combine both translations.
            translationVector_s = toMiddlePoint_s + offsetFromMiddlePoint_s;
        }


        /** Transforms a vector from the PIC system to the beam comoving system.
         *
         * @param currentStep Current simulation step.
         * @param position_s A 3D vector in the PIC coordinate system.
         */
        HDINLINE float3_X coordinateTransform(uint32_t const& currentStep, float3_X const& position_s) const
        {
            float3_X result = position_s - translationVector_s;
            result = Side::rotateSimToBeam(result);
            result[2] -= currentStep * DELTA_T * SPEED_OF_LIGHT;
            return result;
        }

        /** Calculates the probing amplitude at a given position.
         *
         * @param currentStep Current simulation step.
         * @param position_s Position in the simulation coordinate system (x, y, z__at_t_0 - c*t).
         * @returns Probing wave amplitude scaling at position_b.
         */
        HDINLINE float_X operator()(uint32_t const& currentStep, float3_X const& position_s) const
        {
            const float3_X position_b{coordinateTransform(currentStep, position_s)};
            float_X profileFactor = BeamProfile::getFactor(position_b[0], position_b[1]);

            // negative time in front of the pulse positive time behind the pulse
            float_X beamTime = -1.0_X * position_b[2] / SPEED_OF_LIGHT;
            float_X shapeFactor = BeamShape::getFactor(beamTime);

            return profileFactor * shapeFactor;
        }

    private:
        PMACC_ALIGN(translationVector_s, float3_X);
    };
} // namespace picongpu::particles::externalBeam::beam
