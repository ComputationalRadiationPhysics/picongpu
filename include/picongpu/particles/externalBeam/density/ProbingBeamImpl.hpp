/* Copyright 2014-2020 Pawel Ordyna
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

#include "picongpu/plugins/externalBeam/ProbingBeam.hpp"
#include "picongpu/plugins/externalBeam/Side.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace externalBeam
        {
            namespace density
            {
                namespace detail
                {
                    using namespace picongpu::plugins::externalBeam;
                    using namespace picongpu::SI;
                    template<typename T_Side>
                    struct GetCellSurfaceSI
                    {
                        static constexpr float_64 get();
                    };
                    template<>
                    struct GetCellSurfaceSI<XSide>
                    {
                        static constexpr float_64 get()
                        {
                            return CELL_DEPTH_SI * CELL_HEIGHT_SI;
                        }
                    };
                    template<>
                    struct GetCellSurfaceSI<XRSide>
                    {
                        static constexpr float_64 get()
                        {
                            return CELL_DEPTH_SI * CELL_HEIGHT_SI;
                        }
                    };
                    template<>
                    struct GetCellSurfaceSI<YSide>
                    {
                        static constexpr float_64 get()
                        {
                            return CELL_DEPTH_SI * CELL_WIDTH_SI;
                        }
                    };
                    template<>
                    struct GetCellSurfaceSI<YRSide>
                    {
                        static constexpr float_64 get()
                        {
                            return CELL_DEPTH_SI * CELL_WIDTH_SI;
                        }
                    };
                    template<>
                    struct GetCellSurfaceSI<ZSide>
                    {
                        static constexpr float_64 get()
                        {
                            return CELL_WIDTH_SI * CELL_HEIGHT_SI;
                        }
                    };
                    template<>
                    struct GetCellSurfaceSI<ZRSide>
                    {
                        static constexpr float_64 get()
                        {
                            return CELL_WIDTH_SI * CELL_HEIGHT_SI;
                        }
                    };

                } // namespace detail
                template<typename T_ParamClass>
                struct ProbingBeamImpl : public T_ParamClass
                {
                    using ParamClass = T_ParamClass;
                    using ProbingBeam = typename T_ParamClass::ProbingBeam;
                    using Side = typename T_ParamClass::ProbingBeam::Side;
                    static constexpr float_64 photonFluxAtMaxBeamIntensity = ParamClass::photonFluxAtMaxBeamIntensity;
                    static constexpr float_64 CELL_SURFACE_SI = detail::GetCellSurfaceSI<Side>::get();
                    static constexpr float_X PHOTONS_IN_A_CELL
                        = static_cast<float_X>(photonFluxAtMaxBeamIntensity * SI::DELTA_T_SI * CELL_SURFACE_SI);
                    static constexpr float_X REFERENCE_PHOTON_DENSITY = PHOTONS_IN_A_CELL / CELL_VOLUME / BASE_DENSITY;

                    template<typename T_SpeciesType>
                    struct apply
                    {
                        using type = ProbingBeamImpl<ParamClass>;
                    };

                    HINLINE ProbingBeamImpl(uint32_t currentStep)
                        : probingBeam_m()
                        , currentStep_m(currentStep)
                    {
                    }

                    /** Calculate the normalized density
                     *
                     * @param totalCellOffset total offset including all slides [in cells]
                     */
                    HDINLINE float_X operator()(const DataSpace<simDim>& totalCellOffset)
                    {
                        const floatD_X globalCellPos(
                            (precisionCast<float_X>(totalCellOffset) + floatD_X::create(0.5))
                            * cellSize.shrink<simDim>());
                        float3_X position_b = probingBeam_m.coordinateTransform(currentStep_m, globalCellPos);
                        float_X intensity = probingBeam_m(position_b);
                        return intensity * REFERENCE_PHOTON_DENSITY;
                    }

                private:
                    PMACC_ALIGN(probingBeam_m, const ProbingBeam);
                    PMACC_ALIGN(currentStep_m, const uint32_t);
                };
            } // namespace density
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
