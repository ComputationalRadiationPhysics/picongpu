/* Copyright 2021-2023 Brian Marre, Sergei Bastrakov
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

#include <pmacc/traits/HasFlag.hpp>

namespace picongpu::particles::atomicPhysics2
{
    struct SetToAtomicGroundState
    {
        /** set a given ion to its ground state for a given number of bound
         *  electrons
         *
         *  Functor, sets a given ion's boundElectrons attribute,
         *  and, if present, atomicState attribute, to the correct value/s
         *  for its ground state.
         *
         *  Example:
         *    Ar(Z=18), boundElectrons=3 --> super configuration(2,1,0,0,...)
         *
         *  BEWARE: Different Usage from previous SetIonization(),
         *      specify NUMBER of bound electrons NOT charge state.
         *  BEWARE: Uses simple fill from bottom shell wise for ground state
         *      determination, not actually entire truth.
         *  @todo : implement full madelung occupation schema, and exceptions,
         *      like Cu.
         */
        template<typename T_Ion>
        DINLINE void operator()(T_Ion& ion, uint8_t numberBoundElectrons)
        {
            // init atomic state consistently if present
            if constexpr(pmacc::traits::HasFlag<typename T_Ion::FrameType, isAtomicPhysicsIon<>>::type::value)
            {
                // get current Configuration number object
                auto configNumber = ion[atomicConfigNumber_];

                // create blank occupation number vector
                auto occupationNumberVector = pmacc::math::Vector<uint8_t, configNumber.numberLevels>::create(0);
                // could actually be reduced to uint8_t since Z<=98(Californium) for our purposes

                // fill from bottom up until no electrons remaining -> ground state init
                /// @todo : implement Mandelung Schema and exceptions, Brian Marre, 2022
                for(uint8_t level = 1u; level <= configNumber.numberLevels; level++)
                {
                    // g(n) = 2*n^2; for hydrogen like states
                    if(numberBoundElectrons >= 2u * level * level)
                    {
                        (occupationNumberVector)[level - 1u] = 2u * level * level;
                        numberBoundElectrons -= 2u * level * level;
                    }
                    else
                    {
                        (occupationNumberVector)[level - 1u] = numberBoundElectrons;
                        break;
                    }
                }

                // set atomic state index
                ion[atomicConfigNumber_] = configNumber.getAtomicConfigNumber(occupationNumberVector);
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics2
