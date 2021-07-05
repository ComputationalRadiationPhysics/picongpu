/* Copyright 2015-2021 Alexander Grund, Pawel Ordyna
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

#include "picongpu/plugins/photonDetector/accumulation/CountParticles.def"


namespace picongpu
{
    namespace plugins
    {
        namespace photonDetector
        {
            namespace accumulation
            {
                template<typename T_Species>
                HDINLINE acc::CountParticles<T_Species>::CountParticles(
                    uint32_t currentStep,
                    const DetectorParams& detector,
                    DataSpace<simDim>& simSize)
                    : detector_m(detector)
                    , simSize_m(simSize)
                {
                }

                template<typename T_Species>
                template<typename T_Acc, typename T_DetectorBox, typename T_Particle>
                DINLINE void acc::CountParticles<T_Species>::operator()(
                    T_Acc const& acc,
                    T_DetectorBox detectorBox,
                    const DataSpace<DIM2>& targetCellIdx,
                    const T_Particle& particle,
                    const DataSpace<simDim>& globalCellIdx) const
                {
                    const Type amplitude = precisionCast<Type>(particle[weighting_]);
                    cupla::atomicAdd(acc, &(detectorBox(targetCellIdx)), amplitude, alpaka::hierarchy::Blocks{});
                }

                template<typename T_Species>
                HINLINE std::vector<float_64> CountParticles<T_Species>::getUnitDimension()
                {
                    /*
                     */
                    std::vector<float_64> unitDimension(7, 0.0);
                    unitDimension.at(SIBaseUnits::length) = 0.0;
                    unitDimension.at(SIBaseUnits::mass) = 0.0;
                    unitDimension.at(SIBaseUnits::time) = -0.0;
                    unitDimension.at(SIBaseUnits::electricCurrent) = 0.0;
                    return unitDimension;
                }
                template<typename T_Species>
                HDINLINE float_64 CountParticles<T_Species>::getUnit()
                {
                    return 1.0;
                }

                template<typename T_Species>
                HINLINE std::string CountParticles<T_Species>::getName()
                {
                    return "CountParticles";
                }

                template<typename T_Species>
                HINLINE std::string CountParticles<T_Species>::getOpenPMDMeshName()
                {
                    return "particleCount";
                }

                template<typename T_Species>
                HDINLINE acc::CountParticles<T_Species> CountParticles<T_Species>::operator()(
                    uint32_t currentStep,
                    const DetectorParams& detector,
                    DataSpace<simDim>& simSize) const
                {
                    return acc::CountParticles<T_Species>(currentStep, detector, simSize);
                }
            } // namespace accumulation
        } // namespace photonDetector
    } // namespace plugins
} // namespace picongpu
