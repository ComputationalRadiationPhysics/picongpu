/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz
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
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/cuSTL/cursor/BufferCursor.hpp"

namespace pmacc
{
    template<typename TYPE, unsigned DIM>
    class PitchedBox;

    template<typename TYPE>
    class PitchedBox<TYPE, DIM1>
    {
    public:
        enum
        {
            Dim = DIM1
        };
        typedef TYPE ValueType;
        typedef ValueType& RefValueType;
        typedef PitchedBox<TYPE, DIM1> ReducedType;

        HDINLINE RefValueType operator[](const int idx)
        {
            return fixedPointer[idx];
        }

        HDINLINE RefValueType operator[](const int idx) const
        {
            return fixedPointer[idx];
        }

        HDINLINE PitchedBox(TYPE* pointer, const DataSpace<DIM1>& offset, const DataSpace<DIM1>&, const size_t)
            : fixedPointer(pointer + offset[0])
        {
        }

        HDINLINE PitchedBox(TYPE* pointer, const DataSpace<DIM1>& offset) : fixedPointer(pointer + offset[0])
        {
        }

        HDINLINE PitchedBox(TYPE* pointer) : fixedPointer(pointer)
        {
        }

        /*Object must init by copy a valid instance*/
        HDINLINE PitchedBox()
        {
        }

        /*!return the first value in the box (list)
         * @return first value
         */
        HDINLINE RefValueType operator*()
        {
            return *(fixedPointer);
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

    template<typename TYPE>
    class PitchedBox<TYPE, DIM2>
    {
    public:
        enum
        {
            Dim = DIM2
        };
        typedef TYPE ValueType;
        typedef ValueType& RefValueType;
        typedef PitchedBox<TYPE, DIM1> ReducedType;

        HDINLINE PitchedBox(TYPE* pointer, const DataSpace<DIM2>& offset, const DataSpace<DIM2>&, const size_t pitch)
            : pitch(pitch)
            , fixedPointer((TYPE*) ((char*) pointer + offset[1] * pitch) + offset[0])
        {
        }

        HDINLINE PitchedBox(TYPE* pointer, size_t pitch) : pitch(pitch), fixedPointer(pointer)
        {
        }

        /*Object must init by copy a valid instance*/
        HDINLINE PitchedBox()
        {
        }

        HDINLINE ReducedType operator[](const int idx)
        {
            return ReducedType((TYPE*) ((char*) this->fixedPointer + idx * pitch));
        }

        HDINLINE ReducedType operator[](const int idx) const
        {
            return ReducedType((TYPE*) ((char*) this->fixedPointer + idx * pitch));
        }

        HDINLINE PitchedBox(TYPE* pointer, const DataSpace<DIM2>& offset, size_t pitch)
            : pitch(pitch)
            , fixedPointer((TYPE*) ((char*) pointer + offset[1] * pitch) + offset[0])
        {
        }

        /*!return the first value in the box (list)
         * @return first value
         */
        HDINLINE RefValueType operator*()
        {
            return *((TYPE*) fixedPointer);
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
        PMACC_ALIGN(pitch, size_t);
        PMACC_ALIGN(fixedPointer, TYPE*);
    };

    template<typename TYPE>
    class PitchedBox<TYPE, DIM3>
    {
    public:
        enum
        {
            Dim = DIM3
        };
        typedef TYPE ValueType;
        typedef ValueType& RefValueType;
        typedef PitchedBox<TYPE, DIM2> ReducedType;

        HDINLINE ReducedType operator[](const int idx)
        {
            return ReducedType((TYPE*) ((char*) (this->fixedPointer) + idx * pitch2D), pitch);
        }

        HDINLINE ReducedType operator[](const int idx) const
        {
            return ReducedType((TYPE*) ((char*) (this->fixedPointer) + idx * pitch2D), pitch);
        }

        /** constructor
         *
         * @param pointer pointer to the origin of the physical memory
         * @param offset offset (in elements)
         * @param memSize size of the physical memory (in elements)
         * @param pitch number of bytes in one line (first dimension)
         */
        HDINLINE PitchedBox(
            TYPE* pointer,
            const DataSpace<DIM3>& offset,
            const DataSpace<DIM3>& memSize,
            const size_t pitch)
            : pitch(pitch)
            , pitch2D(memSize[1] * pitch)
            , fixedPointer(
                  (TYPE*) ((char*) pointer + offset[2] * (memSize[1] * pitch) + offset[1] * pitch) + offset[0])
        {
        }

        HDINLINE PitchedBox(TYPE* pointer, const size_t pitch, const size_t pitch2D)
            : pitch(pitch)
            , pitch2D(pitch2D)
            , fixedPointer(pointer)
        {
        }

        /*Object must init by copy a valid instance*/
        HDINLINE PitchedBox()
        {
        }

        /*!return the first value in the box (list)
         * @return first value
         */
        HDINLINE RefValueType operator*()
        {
            return *(fixedPointer);
        }

        HDINLINE TYPE const* getPointer() const
        {
            return fixedPointer;
        }
        HDINLINE TYPE* getPointer()
        {
            return fixedPointer;
        }

        HDINLINE pmacc::cursor::BufferCursor<TYPE, DIM3> toCursor() const
        {
            return pmacc::cursor::BufferCursor<TYPE, DIM3>(
                (TYPE*) fixedPointer,
                ::pmacc::math::Size_t<2>(pitch, pitch2D));
        }

    protected:
        HDINLINE PitchedBox<TYPE, DIM2> reduceZ(const int zOffset) const
        {
            return PitchedBox<TYPE, DIM2>((TYPE*) ((char*) (this->fixedPointer) + pitch2D * zOffset), pitch);
        }


        PMACC_ALIGN(pitch, size_t);
        PMACC_ALIGN(pitch2D, size_t);
        PMACC_ALIGN(fixedPointer, TYPE*);
    };


} // namespace pmacc
