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

#include "picongpu/particles/externalBeam/phase/AssingPhase.hpp"
#include "picongpu/particles/PhotonFunctors.hpp"
namespace picongpu
{
    namespace particles
    {
        namespace externalBeam
        {
            namespace initPhase
            {
                template<typename T_ParamClass>
                struct FromSpeciesWavelength
                {
                    using SideCfg = typename T_ParamClass::ProbingBeam::SideCfg;
                    static constexpr float_64 phi0 = T_ParamClass::phi0;
                    PMACC_ALIGN(axisSwap, typename SideCfg::AxisSwapRT);

                    template<typename T_Acc, typename T_MetaData, typename T_Particle, typename... T_Args>
                    DINLINE void operator()(T_Acc const& acc, T_MetaData& meta, T_Particle& particle, T_Args&&...)
                        const
                    {
                        /* this functor will be called only in the first cell (counting from the boundary where
                        the photon beam enters the simulation. So global position along the propagation axis
                        is just the in cell position. */
                        const floatD_X position = particle[position_];
                        // distance from the z_beam=0 plane (the simulation boundary) where the position contribution
                        // to the plane wave phase is 0.
                        const float_X distance{(axisSwap.rotate(position)).z()};
                        const float_X waveNumber{GetAngFrequency<T_Particle>()() / SPEED_OF_LIGHT};
                        const float_X spatialContribution = waveNumber * distance;
                        const float_X phase = spatialContribution + curPhase;

                        AssignPhase<T_Particle>::assign(particle, phase);

                    }
                    template<typename T_Particle, typename T_MetaData>
                    HDINLINE void init(T_MetaData const& meta)
                    {
                        uint32_t currentStep = meta.m_currentStep;
                        curPhase = precisionCast<float_X>(GetPhaseByTimestep<T_Particle>()(currentStep, phi0));
                    }

                private:
                    PMACC_ALIGN(curPhase, float_64);
                };
            } // namespace phase
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
