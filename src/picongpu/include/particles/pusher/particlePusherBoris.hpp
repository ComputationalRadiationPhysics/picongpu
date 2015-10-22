/**
 * Copyright 2013-2015 Heiko Burau, Rene Widera, Richard Pausch
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

#include "types.h"
#include "simulation_defines.hpp"

namespace picongpu
{
namespace particlePusherBoris
{

template<class Velocity, class Gamma>
struct Push
{

    template<typename T_Efield, typename T_Bfield, typename T_Pos, typename T_Mom, typename T_Mass,
             typename T_Charge>
        __host__ DINLINE void operator()(
                                            const T_Bfield bField,
                                            const T_Efield eField,
                                            T_Pos& pos,
                                            T_Mom& mom,
                                            const T_Mass mass,
                                            const T_Charge charge)
    {
        typedef T_Mom MomType;
        typedef T_Bfield BType;

        const float_X QoM = charge / mass;

        const float_X deltaT = DELTA_T;

        const MomType mom_minus = mom + float_X(0.5) * charge * eField * deltaT;

        Gamma gamma;
        const float_X gamma_reci = float_X(1.0) / gamma(mom_minus, mass);
        const float3_X t = float_X(0.5) * QoM * bField * gamma_reci * deltaT;
        const BType s = float_X(2.0) * t * (float_X(1.0) / (float_X(1.0) + math::abs2(t)));

        const MomType mom_prime = mom_minus + math::cross(mom_minus, t);
        const MomType mom_plus = mom_minus + math::cross(mom_prime, s);

        const MomType new_mom = mom_plus + float_X(0.5) * charge * eField * deltaT;
        mom = new_mom;

        Velocity velocity;
        const float3_X vel = velocity(new_mom, mass);

        for(uint32_t d=0;d<simDim;++d)
        {
            pos[d] += (vel[d] * deltaT) / cellSize[d];
        }

    }
};
} //namespace particlePusherBoris
} //namepsace  picongpu
