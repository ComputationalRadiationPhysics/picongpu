/* Copyright 2019-2021 Rene Widera
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

#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/particles/collision/collision.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            //! Functor for the stage of the PIC loop performing particle collision
            class Collision
            {
            public:
                Collision(std::shared_ptr<DeviceHeap>& heap) : m_heap(heap)
                {
                }

                /** Perform particle particle collision
                 *
                 * @param step index of time iteration
                 */
                void operator()(uint32_t const step) const
                {
                    pmacc::meta::ForEach<
                        particles::collision::CollisionPipeline,
                        particles::collision::CallCollider<bmpl::_1>>{}(m_heap, step);
                }

            private:
                std::shared_ptr<DeviceHeap> m_heap;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
