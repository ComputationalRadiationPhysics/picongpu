/**
 * Copyright 2014 Marco Garten
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

namespace picongpu
{
    using namespace PMacc;
    
    /* operations on particles */
    namespace partOp = PMacc::particles::operations;
    
    template<typename T_parentIon, typename T_childElectron, class T_parentFrame, class T_childFrame>
    __device__ void writeElectronIntoFrame(T_parentIon& parentIon,T_childElectron& childElectron,T_parentFrame& parentFrame, T_childFrame& childFrame)
    {
                
        /* each thread sets the multiMask hard on "particle" (=1) and the charge to 1 */
        childElectron[multiMask_] = 1;
        childElectron[chargeState_] = 1;
        const uint32_t weighting = parentIon[weighting_];
//        printf("w: %d ",weighting);
//        printf("cs: %d ",childElectron[chargeState_]);

        /* each thread initializes a clone of the parent ion but leaving out
         * some attributes:
         * - multiMask: because it takes reportedly long to clone
         * - chargeState: because electrons cannot have a charge state other than 1
         * - momentum: because the electron would get a higher energy because of the ion mass */
        PMACC_AUTO(targetElectronClone, partOp::deselect<bmpl::vector3<multiMask, chargeState, momentum> >(childElectron));
        partOp::assign(targetElectronClone, parentIon);
                
        const float_X massIon = getMass<T_parentFrame>(weighting);
//        printf("mI: %e ",massIon);
        const float_X massElectron = getMass<T_childFrame>(weighting);
//        printf("mE: %e ",massElectron);
//        printf("mR: %f",massIon/massElectron); //1836.152588 nice, awesome
//        printf("mom: %e ",parentIon[momentum_].y()); 
        float3_X electronMomentum = float3_X(
        parentIon[momentum_].x( )*(massElectron/massIon),
        parentIon[momentum_].y( )*(massElectron/massIon),
        parentIon[momentum_].z( )*(massElectron/massIon)
                        );
//        printf("mom: %e ",electronMomentum.z());
        childElectron[momentum_] = electronMomentum;
    }
}

