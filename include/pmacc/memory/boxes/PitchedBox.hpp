/* Copyright 2013-2023 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/math/vector/Vector.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    template<typename T_Type, uint32_t T_dim>
    struct PitchedBox
    {
        static constexpr std::uint32_t Dim = T_dim;
        using ValueType = T_Type;
        using RefValueType = ValueType&;

        /** return value the origin pointer is pointing to
         *
         * @return value at the current location
         */
        HDINLINE RefValueType operator*()
        {
            return *this->m_ptr;
        }

        /** get origin pointer
         *
         * @{
         */
        HDINLINE ValueType const* getPointer() const
        {
            return this->m_ptr;
        }

        HDINLINE ValueType* getPointer()
        {
            return this->m_ptr;
        }
        /** @} */

        /*Object must init by copy a valid instance*/
        PitchedBox() = default;

        /** Constructor
         *
         * @tparam T_MemoryIdxType Index type
         * @param pointer pointer to the memory
         * @param extent extent of the memory in elements
         * @param pitch line pitch in bytes
         */
        template<typename T_MemoryIdxType>
        HDINLINE PitchedBox(ValueType* pointer, math::Vector<T_MemoryIdxType, T_dim> const& extent, size_t const pitch)
            : m_ptr(pointer)
        {
            m_pitch.x() = pitch;
            for(uint32_t d = 1; d < T_dim - 1; ++d)
                m_pitch[d] = m_pitch[d - 1u] * static_cast<size_t>(extent[d]);
        }

        PitchedBox(PitchedBox const&) = default;

        /** get value at the given index
         *
         * @tparam T_MemoryIdxType Index type
         * @param idx n-dimensional offset, relative to the origin pointer
         * @return reference to the value
         * @{
         */
        template<typename T_MemoryIdxType>
        HDINLINE ValueType const& operator[](math::Vector<T_MemoryIdxType, T_dim> const& idx) const
        {
            return *ptr(idx);
        }

        template<typename T_MemoryIdxType>
        HDINLINE RefValueType operator[](math::Vector<T_MemoryIdxType, T_dim> const& idx)
        {
            return *const_cast<ValueType*>(ptr(idx));
        }

        /** }@ */

    protected:
        /** get the pointer of the value relative to the origin pointer m_ptr
         *
         * @tparam T_MemoryIdxType Index type
         * @param idx n-dimensional offset
         * @return pointer to value
         */
        template<typename T_MemoryIdxType>
        HDINLINE ValueType const* ptr(math::Vector<T_MemoryIdxType, T_dim> const& idx) const
        {
            /** offset in bytes
             *
             * We calculate the complete offset in bytes even if it would be possible to change the x-dimension with
             * the native types pointer, this is reducing the register footprint.
             */
            size_t offset = sizeof(ValueType) * idx.x();
            for(uint32_t d = 1u; d < T_dim; ++d)
                offset += m_pitch[d - 1u] * idx[d];
            return reinterpret_cast<ValueType const*>(reinterpret_cast<char const*>(this->m_ptr) + offset);
        }

        PMACC_ALIGN(m_ptr, ValueType*);
        PMACC_ALIGN(m_pitch, math::Vector<size_t, T_dim - 1>);
    };

    template<typename T_Type>
    struct PitchedBox<T_Type, DIM1>
    {
        static constexpr std::uint32_t Dim = DIM1;
        using ValueType = T_Type;
        using RefValueType = ValueType&;

        /** return value the origin pointer is pointing to
         *
         * @return value at the current location
         */
        HDINLINE RefValueType operator*()
        {
            return *this->m_ptr;
        }

        /** get origin pointer
         *
         * @{
         */
        HDINLINE ValueType const* getPointer() const
        {
            return this->m_ptr;
        }

        HDINLINE ValueType* getPointer()
        {
            return this->m_ptr;
        }
        /** @} */

        /*Object must init by copy a valid instance*/
        PitchedBox() = default;

        /** Constructor
         *
         * @tparam T_MemoryIdxType Index type
         * @param pointer pointer to the memory
         * @param extent extent of the memory in elements
         * @param pitch line pitch in bytes
         */
        template<typename T_MemoryIdxType>
        HDINLINE PitchedBox(
            ValueType* pointer,
            [[maybe_unused]] math::Vector<T_MemoryIdxType, DIM1> const& extent,
            [[maybe_unused]] size_t const pitch)
            : m_ptr(pointer)
        {
        }

        HDINLINE PitchedBox(ValueType* pointer) : m_ptr(pointer)
        {
        }

        PitchedBox(PitchedBox const&) = default;

        /** get value at the given index
         *
         * @tparam T_MemoryIdxType Index type
         * @param idx offset relative to the origin pointer
         * @return reference to the value
         * @{
         */
        HDINLINE ValueType const& operator[](int const idx) const
        {
            return *(m_ptr + idx);
        }

        HDINLINE RefValueType operator[](int const idx)
        {
            return *(m_ptr + idx);
        }

        HDINLINE ValueType const& operator[](DataSpace<DIM1> const& idx) const
        {
            return *(m_ptr + idx.x());
        }

        HDINLINE RefValueType operator[](DataSpace<DIM1> const& idx)
        {
            return *(m_ptr + idx.x());
        }

        /** @} */

    protected:
        PMACC_ALIGN(m_ptr, ValueType*);
    };
} // namespace pmacc
