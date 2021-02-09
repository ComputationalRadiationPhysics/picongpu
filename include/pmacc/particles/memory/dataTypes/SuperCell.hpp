/* Copyright 2013-2021 Heiko Burau, Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
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
    template<class T_FrameType>
    class SuperCell
    {
    public:
        HDINLINE SuperCell() : firstFramePtr(nullptr), lastFramePtr(nullptr), numParticles(0), mustShiftVal(false)
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

        HDINLINE T_FrameType const* FirstFramePtr() const
        {
            return firstFramePtr;
        }

        HDINLINE T_FrameType const* LastFramePtr() const
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

        HDINLINE uint32_t getSizeLastFrame() const
        {
            constexpr uint32_t frameSize = math::CT::volume<typename T_FrameType::SuperCellSize>::type::value;
            return numParticles ? ((numParticles - 1u) % frameSize + 1u) : 0u;
        }

        HDINLINE uint32_t getNumParticles() const
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
        PMACC_ALIGN(numParticles, uint32_t);
        PMACC_ALIGN(mustShiftVal, bool);
    };

} // namespace pmacc
