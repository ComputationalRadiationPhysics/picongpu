/* Copyright 2019-2023 Brian Marre
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

/** @file
 *
 *This file defines the vector representation of the atomic state
 */

#pragma once

#include <picongpu/particles/flylite/types/Superconfig.hpp>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            namespace stateRepresentation
            {
                template<typename T_Type, uint8_t T_numberStates>
                using LevelVector = flylite::types::Superconfig<T_Type, T_numberStates>;
            } // namespace stateRepresentation
        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
