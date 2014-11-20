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
//        template<class Velocity, class Gamma>
        struct Ionize
        {

            template<typename EType, typename BType, typename PosType, typename MomType, typename MassType, typename ChargeType, typename ChargeStateType >
                    __host__ DINLINE void operator()(
                                                        const BType bField, /* at t=0 */
                                                        const EType eField, /* at t=0 */
                                                        PosType& pos, /* at t=0 */
                                                        MomType& mom, /* at t=-1/2 */
                                                        const MassType mass,
                                                        const ChargeType charge,
                                                        ChargeStateType& chState)
            {
                
                /*Barrier Suppression Ionization for hydrogenlike helium 
                 *charge >= 0 is needed because electrons and ions cannot be 
                 *distinguished, yet.
                 */
//                printf("cs: %d ",chState);
                if (math::abs(eField)*UNIT_EFIELD >= 5.14e7 && chState < 2 && charge >= 0)
                {
                    chState = 1 + chState;
//                    printf("CS: %u ", chState);
                }
                
                /*
                 *int firstIndex = blockIdx.x * blockIdx.y * blockIdx.z * threadIdx.x * threadIdx.y * threadIdx.z;
                 *if (firstIndex == 0)
                 *{
                 *    printf("Charge State: %u", chState);
                 *}
                 */
                
            }
        };
    } //namespace particleIonizerNone
}