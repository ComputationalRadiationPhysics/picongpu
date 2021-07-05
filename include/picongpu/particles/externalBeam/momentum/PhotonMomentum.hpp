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

namespace picongpu
{
    namespace particles
    {
        namespace externalBeam
        {
            namespace momentum
            {
                template<typename T_ParamClass>
                struct PhotonMomentum
                {
                private:
                    using ParamClass = T_ParamClass;
                    using SideCfg = typename T_ParamClass::ProbingBeam::SideCfg;
                    static constexpr float_64 photonMomentum{
                        ParamClass::photonEnergySI / UNIT_ENERGY / static_cast<float_64>(SPEED_OF_LIGHT)};
                    // Photons propagate along the beam z direction
                    // Check if the beam z axis is a reversed pic axis. If so reverse direction.z before axis swap.
                    static constexpr bool reverse = SideCfg::Side::reverse[2];
                    using DirectionBeam =
                        typename boost::mpl::if_c<reverse, mCT::Int<0, 0, -1>, mCT::Int<0, 0, 1>>::type;
                    // Convert into the pic simulation system
                    using DirectionSim = typename SideCfg::AxisSwapCT::template ReverseSwap<DirectionBeam>::type;

                public:
                    template<typename T_Acc, typename T_MetaData, typename T_Particle, typename... T_Args>
                    DINLINE void operator()(T_Acc const& acc, T_MetaData& meta, T_Particle& particle, T_Args&&...)
                        const
                    {
                        float_X const macroWeighting{particle[weighting_]};
                        float3_X const direction{precisionCast<float_X>(DirectionSim::toRT())};
                        // Don't need to normalize direction here since the norm is 1.
                        float3_X const mom{direction * (static_cast<float_X>(photonMomentum) * macroWeighting)};
                        particle[momentum_] = mom;
                    }

                    template<typename T_Particle, typename T_MetaData>
                    HDINLINE void init(T_MetaData const& meta)
                    {
                    }
                };
            } // namespace momentum
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
