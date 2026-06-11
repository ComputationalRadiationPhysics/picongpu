/*
 * SPDX-FileCopyrightText: Rene Widera, Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/DeviceHeap.hpp"

#include <utility>

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
                void operator()(MappingDesc const cellDescription, uint32_t const currentStep) const;

            private:
                std::shared_ptr<DeviceHeap> m_heap;
            };
        } // namespace stage
    } // namespace simulation
} // namespace picongpu
