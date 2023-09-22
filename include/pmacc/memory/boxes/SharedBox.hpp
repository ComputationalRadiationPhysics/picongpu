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

#include "pmacc/math/Vector.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/memory/shared/Allocate.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace detail
    {
        template<typename T_Extent, uint32_t T_dim>
        struct CalcPitch;

        template<typename T_Extent>
        struct CalcPitch<T_Extent, DIM3>
        {
            using type = math::CT::Int<T_Extent::x::value, T_Extent::x::value * T_Extent::y::value>;
        };

        template<typename T_Extent>
        struct CalcPitch<T_Extent, DIM2>
        {
            using type = math::CT::Int<T_Extent::x::value>;
        };
    } // namespace detail

    /** create shared memory on gpu
     *
     * @tparam T_TYPE type of memory objects
     * @tparam T_Extent CT::Vector with size description (per dimension) in elements
     * @tparam T_id unique id for this object
     *              (is needed if more than one instance of shared memory in one kernel is used)
     * @tparam T_dim dimension of the memory (supports DIM1,DIM2 and DIM3)
     */
    template<typename T_Type, typename T_Extent, uint32_t T_id = 0, uint32_t T_dim = T_Extent::dim>
    struct SharedBox
    {
        static constexpr std::uint32_t Dim = T_dim;

        using ValueType = T_Type;
        using RefValueType = ValueType&;
        using Extent = T_Extent;

        HDINLINE
        SharedBox(ValueType* pointer = nullptr) : m_ptr(pointer)
        {
        }

        HDINLINE SharedBox(SharedBox const&) = default;

        /*!return the first value in the box (list)
         * @return first value
         */
        HDINLINE RefValueType operator*()
        {
            return *m_ptr;
        }

        HDINLINE ValueType const* getPointer() const
        {
            return m_ptr;
        }
        HDINLINE ValueType* getPointer()
        {
            return m_ptr;
        }

        /** get value at the given index
         *
         * @tparam T_MemoryIdxType Index type
         * @param idx n-dimansional offset relative to the origin pointer
         * @return reference to the value
         * @{
         */
        template<typename T_MemoryIdxType>
        HDINLINE T_Type const& operator[](math::Vector<T_MemoryIdxType, T_dim> const& idx) const
        {
            return *ptr(idx);
        }

        template<typename T_MemoryIdxType>
        HDINLINE T_Type& operator[](math::Vector<T_MemoryIdxType, T_dim> const& idx)
        {
            return *const_cast<T_Type*>(ptr(idx));
        }

        /** @} */

        /** create a shared memory box
         *
         * This call synchronizes a block and must be called from all threads and
         * not inside a if clauses
         */
        template<typename T_Worker>
        DINLINE static SharedBox init(T_Worker const& worker)
        {
            auto& mem_sh
                = memory::shared::allocate<T_id, memory::Array<ValueType, math::CT::volume<Extent>::type::value>>(
                    worker);
            return {mem_sh.data()};
        }

    private:
        /** get the pointer of the value relative to the origin pointer m_ptr
         *
         * @tparam T_MemoryIdxType Index type
         * @param idx n-dimensional offset
         * @return pointer to value
         */
        HDINLINE T_Type const* ptr(DataSpace<T_dim> const& idx) const
        {
            // offset in elements
            int offset = idx.x();

            if constexpr(Dim >= DIM2)
            {
                using Pitch = typename detail::CalcPitch<Extent, Extent::dim>::type;
                for(uint32_t d = 1u; d < T_dim; ++d)
                    offset += idx[d] * Pitch::toRT()[d - 1];
            }
            return m_ptr + offset;
        }

    protected:
        PMACC_ALIGN(m_ptr, ValueType*);
    };
} // namespace pmacc
