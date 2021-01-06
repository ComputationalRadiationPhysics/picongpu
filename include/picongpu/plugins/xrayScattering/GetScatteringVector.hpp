/* Copyright 2020-2021 Pawel Ordyna
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
#include <pmacc/math/Vector.hpp>

namespace picongpu
{
    namespace plugins
    {
        namespace xrayScattering
        {
            //! Maps a 1D output array index to the corresponding point in the q-space.
            struct GetScatteringVector
            {
                /**
                 * @param q_min Begin of the output range for all axis.
                 * @param q_max End of the output range for all axis.
                 * @param q_step Output array grid spacing.
                 * @param numVectors The output array size.
                 * @param iterOffset Offset for an index shift.
                 */
                HDINLINE GetScatteringVector(
                    float2_X const q_min,
                    float2_X const q_max,
                    float2_X const q_step,
                    DataSpace<DIM2> const numVectors,
                    uint32_t const iterOffset)
                    : m_q_min(q_min)
                    , m_q_max(q_max)
                    , m_q_step(q_step)
                    , m_numVectors(numVectors)
                    , m_iterOffset(iterOffset)
                {
                }

                HDINLINE float2_X operator[](const uint32_t& idx)
                {
                    const uint32_t totalIdx = idx + m_iterOffset;
                    uint32_t i_y(totalIdx % m_numVectors.y());
                    uint32_t i_x(totalIdx / m_numVectors.y());

                    return m_q_min + m_q_step * float2_X(i_x, i_y);
                }

            private:
                // Pmacc struct members memory alignment for objects stored on devices.
                PMACC_ALIGN(m_q_min, const float2_X);
                PMACC_ALIGN(m_q_max, const float2_X);
                PMACC_ALIGN(m_q_step, const float2_X);
                PMACC_ALIGN(m_numVectors, const DataSpace<DIM2>);
                PMACC_ALIGN(m_iterOffset, const uint32_t);
            };
        } // namespace xrayScattering
    } // namespace plugins
} // namespace picongpu
