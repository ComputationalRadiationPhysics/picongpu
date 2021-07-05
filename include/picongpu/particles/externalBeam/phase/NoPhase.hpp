/* Copyright 2021 Pawel Ordyna
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
            namespace initPhase
            {
                struct NoPhase
                {

                    template<typename T_Acc, typename T_MetaData, typename T_Particle, typename... T_Args>
                    DINLINE void
                    operator()(T_Acc const& acc, T_MetaData& meta, T_Particle& particle, T_Args&&...) const
                    {
                    }
                    template<typename T_Particle, typename T_MetaData>
                    HDINLINE void init(T_MetaData const& meta)
                    {
                    }
                };
            } // namespace phase
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
