/* Copyright 2015-2021 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
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
#include "pmacc/particles/memory/dataTypes/Pointer.hpp"


namespace pmacc
{
    /** Wrapper for a raw pointer a PMacc frame
     *
     * @tparam T_Type type of the pointed object
     */
    template<typename T_Type>
    class FramePointer : public Pointer<T_Type>
    {
    private:
        using Base = Pointer<T_Type>;

    public:
        using type = typename Base::type;
        using PtrType = typename Base::PtrType;

        /** default constructor
         *
         * the default pointer points to invalid memory
         */
        HDINLINE FramePointer() : Base()
        {
        }

        HDINLINE FramePointer(PtrType const ptrIn) : Base(ptrIn)
        {
        }

        HDINLINE FramePointer(const Base& other) : Base(other)
        {
        }

        HDINLINE FramePointer(const FramePointer& other) : Base(other)
        {
        }

        HDINLINE FramePointer& operator=(const FramePointer& other)
        {
            Base::operator=(other);
            return *this;
        }

        /** access the Nth particle
         *
         * it is not checked whether `FramePointer` points to valid memory
         *
         * @param idx particle index in the frame
         */
        HDINLINE typename type::ParticleType operator[](const uint32_t idx)
        {
            return (*Base::ptr)[idx];
        }

        /** access the Nth particle
         *
         * it is not checked whether `FramePointer` points to valid memory
         *
         * @param idx particle index in the frame
         */
        HDINLINE const typename type::ParticleType operator[](const uint32_t idx) const
        {
            return (*Base::ptr)[idx];
        }
    };

} // namespace pmacc
