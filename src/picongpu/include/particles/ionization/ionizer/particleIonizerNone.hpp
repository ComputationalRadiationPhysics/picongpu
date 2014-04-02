/* 
 * File:   particleIonizerNone.hpp
 * Author: garten70
 *
 * Created on March 24, 2014, 3:31 PM
 */

#pragma once

#include "types.h"

/* IONIZATION MODEL (formerly PARTICLE PUSHER from pusher/particlePusherFree */

namespace picongpu
{
    namespace particleIonizerNone
    {
        template<class Velocity, class Gamma>
        struct Ionize
        {

            template<typename EType, typename BType, typename PosType, typename MomType, typename MassType, typename ChargeType >
                    __host__ DINLINE void operator()(
                                                        const BType bField,
                                                        const EType eField,
                                                        PosType& pos,
                                                        MomType& mom,
                                                        const MassType mass,
                                                        const ChargeType charge)
            {
                int idx = blockIdx.x*blockIdx.y*blockIdx.z
                if (idx == 0)
                {
                    printf("Ioniiiiiize! \n")
                }

            }
        };
    } //end namespace particleIonizerNone
}