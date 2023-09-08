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
#include "pmacc/types.hpp"

namespace pmacc
{
    template<typename TYPE, unsigned DIM>
    class PitchedBox;

    namespace detail
    {
        template<typename TYPE, unsigned DIM>
        struct PitchedBoxCommon
        {
            static constexpr std::uint32_t Dim = DIM;
            using ValueType = TYPE;
            using RefValueType = ValueType&;

            HDINLINE PitchedBoxCommon(TYPE* p = nullptr) : fixedPointer(p)
            {
            }

            HDINLINE PitchedBoxCommon(PitchedBoxCommon const&) = default;

            /*!return the first value in the box (list)
             * @return first value
             */
            HDINLINE RefValueType operator*()
            {
                return *fixedPointer;
            }

            HDINLINE TYPE const* getPointer() const
            {
                return fixedPointer;
            }

            HDINLINE TYPE* getPointer()
            {
                return fixedPointer;
            }

        protected:
            PMACC_ALIGN(fixedPointer, TYPE*);
        };
    } // namespace detail

    template<typename TYPE>
    class PitchedBox<TYPE, DIM1> : public detail::PitchedBoxCommon<TYPE, DIM1>
    {
        using Base = detail::PitchedBoxCommon<TYPE, DIM1>;

    public:
        using ReducedType = PitchedBox<TYPE, DIM1>;

        /*Object must init by copy a valid instance*/
        HDINLINE PitchedBox() = default;

        HDINLINE PitchedBox(TYPE* pointer) : Base{pointer}
        {
        }

        HDINLINE PitchedBox(PitchedBox const&) = default;

        HDINLINE PitchedBox(TYPE* pointer, const DataSpace<DIM1>& /*memSize*/, const size_t /*pitch*/) : Base(pointer)
        {
        }

        HDINLINE TYPE& operator[](const int idx) const
        {
            return this->fixedPointer[idx];
        }
    };

    template<typename TYPE>
    class PitchedBox<TYPE, DIM2> : public detail::PitchedBoxCommon<TYPE, DIM2>
    {
        using Base = detail::PitchedBoxCommon<TYPE, DIM2>;

    public:
        using ReducedType = PitchedBox<TYPE, DIM1>;

        /*Object must init by copy a valid instance*/
        HDINLINE PitchedBox() = default;

        HDINLINE PitchedBox(TYPE* pointer, size_t pitch) : Base{pointer}, pitch(pitch)
        {
        }

        HDINLINE PitchedBox(TYPE* pointer, const DataSpace<DIM2>& /*memSize*/, const size_t pitch)
            : Base{pointer}
            , pitch(pitch)
        {
        }

        HDINLINE PitchedBox(PitchedBox const&) = default;

        HDINLINE ReducedType operator[](const int idx) const
        {
            return ReducedType((TYPE*) ((char*) this->fixedPointer + idx * pitch));
        }

    protected:
        PMACC_ALIGN(pitch, size_t);
    };

    template<typename TYPE>
    class PitchedBox<TYPE, DIM3> : public detail::PitchedBoxCommon<TYPE, DIM3>
    {
        using Base = detail::PitchedBoxCommon<TYPE, DIM3>;

    public:
        using ReducedType = PitchedBox<TYPE, DIM2>;

        /*Object must init by copy a valid instance*/
        HDINLINE PitchedBox() = default;

        HDINLINE PitchedBox(TYPE* pointer, const size_t pitch, const size_t pitch2D)
            : Base{pointer}
            , pitch(pitch)
            , pitch2D(pitch2D)
        {
        }

        /** constructor
         *
         * @param pointer pointer to the origin of the physical memory
         * @param offset offset (in elements)
         * @param memSize size of the physical memory (in elements)
         * @param pitch number of bytes in one line (first dimension)
         */
        ///@todo(bgruber): is this functionality not provide by DataBox::shift?
        HDINLINE PitchedBox(TYPE* pointer, const DataSpace<DIM3>& memSize, const size_t pitch)
            : Base{pointer}
            , pitch(pitch)
            , pitch2D(memSize[1] * pitch)
        {
        }

        HDINLINE PitchedBox(PitchedBox const&) = default;

        HDINLINE ReducedType operator[](const int idx) const
        {
            return ReducedType((TYPE*) ((char*) (this->fixedPointer) + idx * pitch2D), pitch);
        }

    protected:
        PMACC_ALIGN(pitch, size_t);
        PMACC_ALIGN(pitch2D, size_t);
    };


} // namespace pmacc
