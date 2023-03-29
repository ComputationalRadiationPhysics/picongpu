/* Copyright 2013-2022 Heiko Burau, Rene Widera, Benjamin Worpitz
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
        template<typename T_Vector, typename T_TYPE>
        HDINLINE auto& subscript(T_TYPE* p, int const idx, std::integral_constant<uint32_t, 1>)
        {
            return p[idx];
        }

        template<typename T_Vector, typename T_TYPE>
        HDINLINE auto* subscript(T_TYPE* p, int const idx, std::integral_constant<uint32_t, 2>)
        {
            return p + idx * T_Vector::x::value;
        }

        template<typename T_Vector, typename T_TYPE>
        HDINLINE auto* subscript(T_TYPE* p, int const idx, std::integral_constant<uint32_t, 3>)
        {
            return p + idx * (T_Vector::x::value * T_Vector::y::value);
        }
    } // namespace detail

    /** create shared memory on gpu
     *
     * @tparam T_TYPE type of memory objects
     * @tparam T_Vector CT::Vector with size description (per dimension)
     * @tparam T_id unique id for this object
     *              (is needed if more than one instance of shared memory in one kernel is used)
     * @tparam T_dim dimension of the memory (supports DIM1,DIM2 and DIM3)
     */
    template<typename T_TYPE, typename T_Vector, uint32_t T_id = 0, uint32_t T_dim = T_Vector::dim>
    struct SharedBox
    {
        static constexpr std::uint32_t Dim = T_dim;

        using ValueType = T_TYPE;
        using RefValueType = ValueType&;
        using Size = T_Vector;

        HDINLINE
        SharedBox(ValueType* pointer = nullptr) : fixedPointer(pointer)
        {
        }

        HDINLINE SharedBox(SharedBox const&) = default;

        using ReducedType1D = T_TYPE&;
        using ReducedType2D = SharedBox<T_TYPE, typename math::CT::shrinkTo<T_Vector, 1>::type, T_id>;
        using ReducedType3D = SharedBox<T_TYPE, typename math::CT::shrinkTo<T_Vector, 2>::type, T_id>;
        using ReducedType
            = std::conditional_t<Dim == 1, ReducedType1D, std::conditional_t<Dim == 2, ReducedType2D, ReducedType3D>>;

        HDINLINE ReducedType operator[](const int idx) const
        {
            ///@todo(bgruber): inline and replace this by if constexpr in C++17
            return {detail::subscript<T_Vector>(fixedPointer, idx, std::integral_constant<uint32_t, T_dim>{})};
        }

        /*!return the first value in the box (list)
         * @return first value
         */
        HDINLINE RefValueType operator*()
        {
            return *fixedPointer;
        }

        HDINLINE ValueType const* getPointer() const
        {
            return fixedPointer;
        }
        HDINLINE ValueType* getPointer()
        {
            return fixedPointer;
        }

        /** create a shared memory box
         *
         * This call synchronizes a block and must be called from all threads and
         * not inside a if clauses
         */
        template<typename T_Worker>
        static DINLINE SharedBox init(T_Worker const& worker)
        {
            auto& mem_sh
                = memory::shared::allocate<T_id, memory::Array<ValueType, math::CT::volume<Size>::type::value>>(
                    worker);
            return {mem_sh.data()};
        }

    protected:
        PMACC_ALIGN(fixedPointer, ValueType*);
    };
} // namespace pmacc
