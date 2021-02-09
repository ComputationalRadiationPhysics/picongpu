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
#include "picongpu/plugins/xrayScattering/beam/Side.hpp"
#include "picongpu/plugins/xrayScattering/beam/SecondaryRotation.hpp"
#include "picongpu/param/xrayScattering.param"

namespace picongpu
{
    namespace plugins
    {
        namespace xrayScattering
        {
            namespace beam
            {
                //! Get the global domain size as a 3D vector in 3D and 2D simulations.
                template<unsigned DIM>
                HINLINE float3_X getDomainSize();

                // For 3D simulations:
                template<>
                HINLINE float3_X getDomainSize<DIM3>()
                {
                    DataSpace<DIM3> globalDomainSize = Environment<DIM3>::get().SubGrid().getGlobalDomain().size;
                    return precisionCast<float_X>(globalDomainSize);
                }

                // For 2D simulations:
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
                template<typename T_Side, typename T_SecondaryRotation>
                struct CoordinateTransform
                {
                    using Side = T_Side;
                    using SecondaryRotation = T_SecondaryRotation;


                    HINLINE CoordinateTransform()
                    {
                        // TODO: Fix the translation in the coordinate transform. The
                        //  position in the beam system is wrongly calculated.
                        //  Orientation is correct.
                        /*
                        using namespace picongpu::plugins::xrayScattering::beam;
                        // Find the coordinate system translation:
                        // Starting in the beam coordinate system.
                        // Transverse(to the beam propagation direction) offset from the
                        // initial position (the middle of the simulation box side).
                        float2_X offsetTrans_b
                        {
                            BEAM_OFFSET[ 0 ] / UNIT_LENGTH,
                            BEAM_OFFSET[ 1 ] / UNIT_LENGTH
                        };
                        // Offset along the propagation direction, defined by the beam
                        // delay.
                        float_X offsetParallel_b = beamDelay_SI / UNIT_TIME *
                            SPEED_OF_LIGHT;
                        // Complete offset from the initial position.
                        float3_X offsetFromMiddlePoint_b
                            {
                                offsetTrans_b[ 0 ],
                                offsetTrans_b[ 1 ],
                                -1 * offsetParallel_b
                            };

                         // Move to the PIC coordinate system.
                         offsetFromMiddlePoint_b = SecondaryRotation::ReverseOperation::
                            rotate( offsetFromMiddlePoint_b );
                        float3_X offsetFromMiddlePoint_s = Side::FirstRotation::reverse(
                            offsetFromMiddlePoint_b );

                        // Find the initial position in the PIC coordinate system.
                        float3_X toMiddlePoint_s = cellSize * getDomainSize< simDim >( );
                        for ( uint32_t ii = 0; ii < 3; ii++ )
                        {
                            toMiddlePoint_s[ ii ] *= Side::beamStartPosition[ ii ];
                        }
                        // Combine both translations.
                        translationVector_s =  toMiddlePoint_s + offsetFromMiddlePoint_s;
                        */
                    }


                    /** Transforms a vector from the PIC system to the beam comoving system.
                     *
                     * @param currentStep Current simulation step.
                     * @param position_s A 3D vector in the PIC coordinate system.
                     */
                    HDINLINE float3_X operator()(uint32_t const& currentStep, float3_X const& position_s)
                    {
                        // TODO: Uncomment after fixing the translation.
                        float3_X result = position_s; /* - translationVector_s;
                        result[ 2 ] -= currentStep * DELTA_T * SPEED_OF_LIGHT;
                        */
                        result = Side::FirstRotation::rotate(result);
                        result = SecondaryRotation::rotate(result);
                        return result;
                    }


                    //! Wrapper for 2D vectors.
                    HDINLINE float3_X operator()(uint32_t const& currentStep, float2_X const& position_s)
                    {
                        float3_X pos{position_s[0], position_s[1], 0.0};
                        return (*this)(currentStep, std::move(pos));
                    }


                private:
                    // TODO: Uncomment after fixing the translation.
                    // PMACC_ALIGN( translationVector_s, float3_X );
                };
            } // namespace beam
        } // namespace xrayScattering
    } // namespace plugins
} // namespace picongpu
