/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/particles/memory/dataTypes/Pointer.hpp"
#include "pmacc/types.hpp"

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

        HDINLINE FramePointer(Base const& other) : Base(other)
        {
        }

        HDINLINE FramePointer(FramePointer const& other) : Base(other)
        {
        }

        HDINLINE FramePointer& operator=(FramePointer const& other)
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
        HDINLINE typename type::ParticleType operator[](uint32_t const idx)
        {
            return (*Base::ptr)[idx];
        }

        /** access the Nth particle
         *
         * it is not checked whether `FramePointer` points to valid memory
         *
         * @param idx particle index in the frame
         */
        HDINLINE const typename type::ParticleType operator[](uint32_t const idx) const
        {
            return (*Base::ptr)[idx];
        }
    };

} // namespace pmacc
