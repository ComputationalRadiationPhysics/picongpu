/* Copyright 2013-2023 Heiko Burau, Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"

#include "pmacc/math/vector/compile-time/Vector.hpp"


namespace pmacc
{
    template<typename T_FrameType, typename T_SuperCellSize>
    class SuperCell
    {
    public:
        using SuperCellSize = T_SuperCellSize;

        HDINLINE SuperCell() : firstFramePtr(nullptr), lastFramePtr(nullptr)
        {
        }

        HDINLINE T_FrameType* FirstFramePtr()
        {
            return firstFramePtr;
        }

        HDINLINE T_FrameType* LastFramePtr()
        {
            return lastFramePtr;
        }

        HDINLINE T_FrameType* FirstFramePtr() const
        {
            return firstFramePtr;
        }

        HDINLINE T_FrameType* LastFramePtr() const
        {
            return lastFramePtr;
        }

        HDINLINE bool mustShift() const
        {
            return mustShiftVal;
        }

        HDINLINE void setMustShift(bool const value)
        {
            mustShiftVal = value;
        }

        //! get number of particle in the last frame
        HDINLINE uint32_t getSizeLastFrame() const
#if(ALPAKA_ACC_GPU_HIP_ENABLED && (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR) == 502)
            /* ROCm 5.2.0 producing particle loss in KernelShiftParticles if this method is defined as `const`.
             * see: https://github.com/ComputationalRadiationPhysics/picongpu/issues/4305
             */
            volatile
#endif
        {
            constexpr uint32_t frameSize = T_FrameType::frameSize;

            /* NOTE on result expression understanding:
             * (numParticles % frameSize) =^= how many particle did not fit in a full frame?
             *
             * but we need how many are in the last frame,
             * => (numParticles - 1u) % frameSize + 1u
             *   only shift by one which is reversed by + 1u
             * => will return the same result for numParticles =/= i * frameSize ;i \in N
             * and for numParticles == i * frameSize, i \in N it will return
             *  ((frameSize * i) - 1u) % frameSize + 1u = (frameSize - 1u) + 1u = frameSize
             */
            // avoids underflow for uint32_t numParticles = 0u
            return numParticles ? ((numParticles - 1u) % frameSize + 1u) : 0u;
        }

        HDINLINE uint32_t getNumParticles() const
#if(ALPAKA_ACC_GPU_HIP_ENABLED && (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR) == 502)
            /* ROCm 5.2.0 producing particle loss in KernelShiftParticles if this method is defined as `const`.
             * see: https://github.com/ComputationalRadiationPhysics/picongpu/issues/4305
             */
            volatile
#endif
        {
            return numParticles;
        }

        HDINLINE void setNumParticles(uint32_t const size)
        {
            numParticles = size;
        }

    public:
        PMACC_ALIGN(firstFramePtr, T_FrameType*);
        PMACC_ALIGN(lastFramePtr, T_FrameType*);

    private:
        PMACC_ALIGN(numParticles, uint32_t){0};
        PMACC_ALIGN(mustShiftVal, bool){false};
    };

} // namespace pmacc
