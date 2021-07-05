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

#include "picongpu/plugins/externalBeam/AxisSwap.hpp"
#include "picongpu/plugins/externalBeam/CoordinateTransform.def"
#include "picongpu/plugins/externalBeam/SecondaryRotation.hpp"
#include "picongpu/plugins/externalBeam/Side.hpp"

#include <pmacc/preprocessor/struct.hpp>
#include <utility>

namespace picongpu
{
    namespace plugins
    {
        namespace externalBeam
        {
            using namespace picongpu;
            //! Get the global domain size as a 3D vector in 3D and 2D simulations.
            template<unsigned DIM>
            HINLINE float3_X getDomainSize();
            // For 3D simulations:
            template<>
            HINLINE float3_X getDomainSize<DIM3>()
            {
                DataSpace<DIM3> globalDomainSize = Environment<DIM3>::get().SubGrid().getGlobalDomain().size;
                return precisionCast<float_X>(globalDomainSize);
            } // For 2D simulations:
            template<>
            HINLINE float3_X getDomainSize<DIM2>()
            {
                auto globalDomainSize = Environment<DIM2>::get().SubGrid().getGlobalDomain().size;
                return float3_X(globalDomainSize[0], globalDomainSize[1], 0.0);
            }

            /** Defines a coordinate transform from the PIC system into the beam system.
             *
             * @tparam T_Side Side from which the probing beam is shot at the target.
             * @tparam T_SecondaryRotation Rotation of the beam propagation direction.
             */
            template<
                typename T_Side,
                typename T_SecondaryRotation,
                typename T_OffsetParam>
            struct CoordinateTransform
            {
                using Side = T_Side;
                using SideCfg = ProbingSideCfg<T_Side>;
                using SecondaryRotation = T_SecondaryRotation;
                using OffsetParam = T_OffsetParam;


            private:
                PMACC_ALIGN(firstRotation, typename SideCfg::AxisSwapRT);
                PMACC_ALIGN(translationVector_s, float3_X);

            public:
                HINLINE CoordinateTransform() : firstRotation()
                {
                    using namespace picongpu::plugins::externalBeam;

                    // Find the coordinate system translation:
                    // Starting in the beam coordinate system.
                    // Transverse(to the beam propagation direction) offset from the
                    // initial position (the middle of the simulation box side).
                    float2_X offsetTrans_b = OffsetParam().beamOffset_SI / UNIT_LENGTH;
                    // Offset along the propagation direction, defined by the beam
                    // delay.
                    float_X beamDelay = OffsetParam::beamDelay_SI / UNIT_TIME;
                    float_X offsetParallel_b = beamDelay * SPEED_OF_LIGHT;
                    // Complete offset from the initial position.
                    float3_X offsetFromMiddlePoint_b{offsetTrans_b[0], offsetTrans_b[1], -1 * offsetParallel_b};

                    // Move to the PIC coordinate system.
                    offsetFromMiddlePoint_b = SecondaryRotation::ReverseOperation::rotate(offsetFromMiddlePoint_b);
                    float3_X offsetFromMiddlePoint_s = firstRotation.reverse(offsetFromMiddlePoint_b);

                    // Find the initial position in the PIC coordinate system.
                    float3_X toMiddlePoint_s = cellSize * getDomainSize<simDim>();
                    for(uint32_t ii = 0; ii < 3; ii++)
                    {
                        toMiddlePoint_s[ii] *= SideCfg::Side::beamStartPosition[ii];
                    }
                    // Combine both translations.
                    translationVector_s = toMiddlePoint_s + offsetFromMiddlePoint_s;
                }


                /** Transforms a vector from the PIC system to the beam comoving system.
                 *
                 * @param currentStep Current simulation step.
                 * @param position_s A 3D vector in the PIC coordinate system.
                 */
                HDINLINE float3_X operator()(uint32_t const& currentStep, float3_X const& position_s) const
                {
                    // TODO: Uncomment after fixing the translation.
                    float3_X result = position_s - translationVector_s;
                    result = firstRotation.rotate(result);
                    result = SecondaryRotation::rotate(result);
                    result[2] -= currentStep * DELTA_T * SPEED_OF_LIGHT;
                    return result;
                }

                //! Wrapper for 2D vectors.
                HDINLINE float3_X operator()(uint32_t const& currentStep, float2_X const& position_s) const
                {
                    float3_X pos{position_s[0], position_s[1], 0.0};
                    return (*this) (currentStep, std::move(pos));
                }
            };

        } // namespace externalBeam
    } // namespace plugins
} // namespace picongpu
